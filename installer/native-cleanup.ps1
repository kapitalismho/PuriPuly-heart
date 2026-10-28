param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$Plan,
    [Parameter(Mandatory = $true)][string]$ResultFile
)

$ErrorActionPreference = 'Stop'

try {
    Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.RegularExpressions;
using Microsoft.Win32.SafeHandles;

public static class NativeLegacyCleanup
{
    const uint ReadAttributes = 0x80;
    const uint GenericRead = 0x80000000;
    const uint DeleteAccess = 0x10000;
    const uint ShareRead = 1;
    const uint OpenExisting = 3;
    const uint OpenReparsePoint = 0x00200000;
    const uint BackupSemantics = 0x02000000;
    const uint ReparsePoint = 0x400;
    const uint DirectoryAttribute = 0x10;

    [StructLayout(LayoutKind.Sequential)]
    struct FileInfo
    {
        public uint Attributes;
        public System.Runtime.InteropServices.ComTypes.FILETIME Creation;
        public System.Runtime.InteropServices.ComTypes.FILETIME Access;
        public System.Runtime.InteropServices.ComTypes.FILETIME Write;
        public uint Volume;
        public uint SizeHigh;
        public uint SizeLow;
        public uint Links;
        public uint IndexHigh;
        public uint IndexLow;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct Disposition
    {
        [MarshalAs(UnmanagedType.U1)] public bool Delete;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern SafeFileHandle CreateFile(string path, uint access, uint sharing, IntPtr security, uint creation, uint flags, IntPtr template);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool GetFileInformationByHandle(SafeFileHandle handle, out FileInfo info);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetFileInformationByHandle(SafeFileHandle handle, int kind, ref Disposition info, uint bytes);

    public static string Relative(string value)
    {
        if (String.IsNullOrEmpty(value)) throw new IOException("Empty owned path");
        string normalized = value.Replace('/', '\\');
        foreach (string part in normalized.Split('\\'))
        {
            if (part.Length == 0 || part == "." || part == ".." || part.EndsWith(".") || part.EndsWith(" ") ||
                Regex.IsMatch(part, "[\\x00-\\x1f<>:\"|?*]") ||
                Regex.IsMatch(part, "^(CON|CONIN\\$|CONOUT\\$|CLOCK\\$|PRN|AUX|NUL|COM[0-9\u00b9\u00b2\u00b3]|LPT[0-9\u00b9\u00b2\u00b3])( *\\.| *$)", RegexOptions.IgnoreCase))
                throw new IOException("Unsafe owned path: " + value);
        }
        return normalized;
    }

    public static bool Protected(string value)
    {
        string root = Relative(value).Split('\\')[0];
        string stem = root.Split('.')[0];
        return Regex.IsMatch(stem, "^(settings|secrets|prompts|extensions|models)$", RegexOptions.IgnoreCase) ||
            String.Equals(root, ".env", StringComparison.OrdinalIgnoreCase) ||
            root.StartsWith(".env.", StringComparison.OrdinalIgnoreCase);
    }

    public static string TargetRoot(string value)
    {
        if (!Regex.IsMatch(value, @"^[A-Za-z]:[\\/]")) throw new IOException("Install root must be a local absolute drive path");
        string full = Path.GetFullPath(value).TrimEnd('\\');
        if (full.Length <= 3 || !String.Equals(full, value.TrimEnd('\\'), StringComparison.OrdinalIgnoreCase))
            throw new IOException("Install root must be a normalized non-root directory");
        return full;
    }

    static SafeFileHandle Open(string path, uint access, bool directory, out bool absent)
    {
        SafeFileHandle handle = CreateFile(path, access, ShareRead, IntPtr.Zero, OpenExisting, OpenReparsePoint | BackupSemantics, IntPtr.Zero);
        absent = false;
        if (handle.IsInvalid)
        {
            int error = Marshal.GetLastWin32Error();
            handle.Dispose();
            if (error == 2 || error == 3) { absent = true; return null; }
            throw new IOException(path + ": " + new Win32Exception(error).Message);
        }
        try
        {
            FileInfo info;
            if (!GetFileInformationByHandle(handle, out info)) throw new Win32Exception(Marshal.GetLastWin32Error());
            if ((info.Attributes & ReparsePoint) != 0) throw new IOException("Link or reparse point rejected: " + path);
            if (((info.Attributes & DirectoryAttribute) != 0) != directory) throw new IOException("Unexpected file/directory type: " + path);
            if (!directory && info.Links != 1) throw new IOException("Hard-linked legacy file rejected: " + path);
            return handle;
        }
        catch { handle.Dispose(); throw; }
    }

    static bool LockAncestors(string path, List<SafeFileHandle> locks)
    {
        string volume = Path.GetPathRoot(path);
        string current = volume;
        string[] parts = path.Substring(volume.Length).Split(new char[] { '\\' }, StringSplitOptions.RemoveEmptyEntries);
        for (int i = -1; i < parts.Length; i++)
        {
            if (i >= 0) current = Path.Combine(current, parts[i]);
            bool absent;
            SafeFileHandle handle = Open(current, ReadAttributes, true, out absent);
            if (absent) return false;
            locks.Add(handle);
        }
        return true;
    }

    static void Unlock(List<SafeFileHandle> locks)
    {
        for (int i = locks.Count - 1; i >= 0; i--) locks[i].Dispose();
    }

    public static void CheckRoot(string root)
    {
        var locks = new List<SafeFileHandle>();
        try { LockAncestors(root, locks); }
        finally { Unlock(locks); }
    }

    public sealed class Snapshot
    {
        public string Text;
        public long Bytes;
        public string Sha256;
    }

    public static Snapshot ReadSnapshot(string root, string relative, string expectedSha256)
    {
        string path = Path.Combine(root, Relative(relative));
        var locks = new List<SafeFileHandle>();
        try
        {
            if (!LockAncestors(Path.GetDirectoryName(path), locks)) return null;
            bool absent;
            using (SafeFileHandle handle = Open(path, GenericRead, false, out absent))
            {
                if (absent) return null;
                using (var stream = new FileStream(handle, FileAccess.Read))
                using (var hash = SHA256.Create())
                {
                    long bytes = stream.Length;
                    string digest = BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
                    if (!String.IsNullOrEmpty(expectedSha256) && !String.Equals(digest, expectedSha256, StringComparison.Ordinal))
                        throw new IOException("Ownership manifest SHA256 differs");
                    stream.Position = 0;
                    using (var reader = new StreamReader(stream, new System.Text.UTF8Encoding(false, true)))
                        return new Snapshot { Text = reader.ReadToEnd(), Bytes = bytes, Sha256 = digest };
                }
            }
        }
        finally { Unlock(locks); }
    }

    public static bool File(string root, string relative, long bytes, string sha256)
    {
        if (bytes < 0) throw new IOException("Invalid owned file size");
        return RemoveFile(root, relative, bytes, true, sha256);
    }

    public static bool FileByHash(string root, string relative, string sha256)
    {
        return RemoveFile(root, relative, 0, false, sha256);
    }

    static bool RemoveFile(string root, string relative, long bytes, bool checkSize, string sha256)
    {
        relative = Relative(relative);
        if (!Regex.IsMatch(sha256, "^[0-9a-f]{64}$")) throw new IOException("Invalid owned file identity");
        string path = Path.Combine(root, relative);
        var locks = new List<SafeFileHandle>();
        try
        {
            if (!LockAncestors(Path.GetDirectoryName(path), locks)) return false;
            bool absent;
            using (SafeFileHandle handle = Open(path, GenericRead | DeleteAccess, false, out absent))
            {
                if (absent) return false;
                using (var stream = new FileStream(handle, FileAccess.Read))
                using (var hash = SHA256.Create())
                {
                    if (checkSize && stream.Length != bytes) throw new IOException("Modified legacy file (size differs): " + path);
                    string actual = BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
                    if (!String.Equals(actual, sha256, StringComparison.Ordinal)) throw new IOException("Modified legacy file (SHA256 differs): " + path);
                    var disposition = new Disposition { Delete = true };
                    if (!SetFileInformationByHandle(handle, 4, ref disposition, 1))
                        throw new IOException(path + ": " + new Win32Exception(Marshal.GetLastWin32Error()).Message);
                    return true;
                }
            }
        }
        finally { Unlock(locks); }
    }

    public static void EmptyDirectory(string root, string relative)
    {
        string path = Path.Combine(root, Relative(relative));
        var locks = new List<SafeFileHandle>();
        try
        {
            if (!LockAncestors(Path.GetDirectoryName(path), locks)) return;
            bool absent;
            using (SafeFileHandle handle = Open(path, ReadAttributes | DeleteAccess, true, out absent))
            {
                if (absent) return;
                var disposition = new Disposition { Delete = true };
                if (!SetFileInformationByHandle(handle, 4, ref disposition, 1))
                {
                    int error = Marshal.GetLastWin32Error();
                    if (error != 145) throw new IOException(path + ": " + new Win32Exception(error).Message);
                }
            }
        }
        finally { Unlock(locks); }
    }
}
'@
    $document = Get-Content -LiteralPath $Plan -Raw -Encoding UTF8 | ConvertFrom-Json
    if (($document.schema -ne 'puripuly-heart/native-installer-cleanup/v1') -or
        ($document.inventory -isnot [Array]) -or
        ([string]$document.native_manifest_sha256 -cnotmatch '^[0-9a-f]{64}$') -or
        ([string]$document.legacy_manifest_sha256 -cnotmatch '^[0-9a-f]{64}$')) {
        throw 'Invalid cleanup plan schema or manifest identity'
    }
    $target = [NativeLegacyCleanup]::TargetRoot($Root)
    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $expectedDirectories = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($item in $document.inventory) {
        if ([NativeLegacyCleanup]::Protected([string]$item.path)) { throw "Protected user data is not a cleanup candidate: $($item.path)" }
        $relative = [NativeLegacyCleanup]::Relative([string]$item.path)
        if (!$seen.Add($relative)) { throw "Duplicate owned path: $relative" }
        if (($item.bytes -isnot [int]) -and ($item.bytes -isnot [long])) { throw "Invalid owned size: $relative" }
        if (($item.bytes -lt 0) -or ([string]$item.sha256 -cnotmatch '^[0-9a-f]{64}$')) { throw "Invalid owned identity: $relative" }
        $ancestor = [IO.Path]::GetDirectoryName($relative)
        while ($ancestor) {
            [void]$expectedDirectories.Add($ancestor)
            $ancestor = [IO.Path]::GetDirectoryName($ancestor)
        }
    }
    foreach ($directory in $expectedDirectories) {
        if ($seen.Contains($directory)) { throw "Owned file is also a directory ancestor: $directory" }
    }
    $directories = @($expectedDirectories | Sort-Object @{Expression = { ($_ -split '\\').Count }; Descending = $true}, @{Expression = { $_ }})
    [NativeLegacyCleanup]::CheckRoot($target)
    $oldRuntime = [NativeLegacyCleanup]::ReadSnapshot($target, 'gpu-worker-runtime.json', $null)
    $oldRuntimeFiles = @()
    if ($null -ne $oldRuntime) {
        $currentSnapshot = [NativeLegacyCleanup]::ReadSnapshot($target, 'native-artifact-manifest.json', [string]$document.native_manifest_sha256)
        if ($null -eq $currentSnapshot) { throw 'Current native artifact ownership manifest is missing' }
        $current = $currentSnapshot.Text | ConvertFrom-Json
        if (($current.schema -ne 'puripuly-heart/native-artifact-manifest/v1') -or ($current.inventory -isnot [Array])) {
            throw 'Invalid current native artifact ownership manifest'
        }
        $currentPaths = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
        foreach ($item in $current.inventory) {
            [void]$currentPaths.Add([NativeLegacyCleanup]::Relative([string]$item.path))
        }
        $prior = $oldRuntime.Text | ConvertFrom-Json
        if (($prior.schema_version -ne 1) -or ($prior.cpu_selection -cne 'upstream-cpuid') -or
            ([string]$prior.build_id -cnotmatch '^[0-9a-f]{32}$') -or ($prior.files -isnot [Array])) {
            throw 'Invalid retired GPU worker ownership inventory'
        }
        $priorPaths = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
        foreach ($item in $prior.files) {
            $name = [NativeLegacyCleanup]::Relative([string]$item.path)
            if (($name -cnotmatch '^(PuriPulyHeartGpuWorker\.exe|transcribe\.dll|ggml\.dll|ggml-base\.dll|ggml-vulkan\.dll|ggml-cpu-[a-z0-9][a-z0-9_-]*\.dll)$') -or
                ([string]$item.sha256 -cnotmatch '^[0-9a-f]{64}$') -or !$priorPaths.Add($name)) {
                throw 'Invalid retired GPU worker ownership entry'
            }
            if (!$currentPaths.Contains($name)) { $oldRuntimeFiles += $item }
        }
        foreach ($required in @('PuriPulyHeartGpuWorker.exe', 'transcribe.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-vulkan.dll', 'ggml-cpu-x64.dll')) {
            if (!$priorPaths.Contains($required)) { throw 'Retired GPU worker ownership inventory is incomplete' }
        }
        if (@($priorPaths | Where-Object { $_ -cmatch '^ggml-cpu-' }).Count -lt 2) {
            throw 'Retired GPU worker optimized CPU ownership inventory is incomplete'
        }
    }
    $removed = 0
    $preserved = New-Object 'System.Collections.Generic.List[string]'
    foreach ($item in $document.inventory) {
        try {
            if ([NativeLegacyCleanup]::File($target, [string]$item.path, [long]$item.bytes, [string]$item.sha256)) { $removed++ }
        } catch {
            $preserved.Add($_.Exception.GetBaseException().Message)
        }
    }
    $oldRuntimeFailed = $false
    foreach ($item in $oldRuntimeFiles) {
        try {
            if ([NativeLegacyCleanup]::FileByHash($target, [string]$item.path, [string]$item.sha256)) { $removed++ }
        } catch {
            $oldRuntimeFailed = $true
            $preserved.Add($_.Exception.GetBaseException().Message)
        }
    }
    if (($null -ne $oldRuntime) -and !$oldRuntimeFailed) {
        try {
            if ([NativeLegacyCleanup]::File($target, 'gpu-worker-runtime.json', $oldRuntime.Bytes, $oldRuntime.Sha256)) { $removed++ }
        } catch {
            $preserved.Add($_.Exception.GetBaseException().Message)
        }
    }
    foreach ($directory in $directories) {
        try {
            [NativeLegacyCleanup]::EmptyDirectory($target, $directory)
        } catch {
            $preserved.Add($_.Exception.GetBaseException().Message)
        }
    }
    [IO.File]::WriteAllText($ResultFile, ("Removed legacy files: {0}; preserved inaccessible or changed paths: {1}.{2}{3}" -f $removed, $preserved.Count, [Environment]::NewLine, ($preserved -join [Environment]::NewLine)), (New-Object Text.UTF8Encoding($false)))
    exit 0
}
catch {
    [IO.File]::WriteAllText($ResultFile, $_.Exception.GetBaseException().Message, (New-Object Text.UTF8Encoding($false)))
    exit 20
}
