using System;
using System.Collections;
using System.Diagnostics;
using System.IO;

class Launcher
{
    static int Main(string[] args)
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        string dataDir = Path.Combine(root, "data");
        string pythonExe = Path.Combine(root, "python", "python.exe");
        string appSrc = Path.Combine(root, "app", "src");
        string mainPy = Path.Combine(appSrc, "puripuly_heart", "main.py");

        if (!File.Exists(pythonExe))
        {
            Console.Error.WriteLine("python.exe not found in: " + Path.Combine(root, "python"));
            return 1;
        }

        if (!File.Exists(mainPy))
        {
            Console.Error.WriteLine("main.py not found in: " + Path.Combine(appSrc, "puripuly_heart"));
            return 1;
        }

        Directory.CreateDirectory(dataDir);

        var psi = new ProcessStartInfo
        {
            FileName = pythonExe,
            UseShellExecute = false,
            WorkingDirectory = root,
        };

        foreach (DictionaryEntry entry in Environment.GetEnvironmentVariables())
        {
            psi.Environment[(string)entry.Key] = (string)entry.Value;
        }
        psi.Environment["PURIPULY_HEART_DATA_DIR"] = dataDir;

        string cmdArgs = "\"" + mainPy + "\"";
        foreach (string arg in args)
        {
            cmdArgs += " \"" + arg + "\"";
        }
        psi.Arguments = cmdArgs;

        try
        {
            using (var proc = Process.Start(psi))
            {
                proc.WaitForExit();
                return proc.ExitCode;
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("Failed to start: " + ex.Message);
            return 1;
        }
    }
}
