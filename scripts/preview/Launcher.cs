using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class Launcher
{
    [STAThread]
    private static void Main()
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        try
        {
            var start = new ProcessStartInfo(Path.Combine(root, "runtime", "pythonw.exe"));
            start.Arguments = "\"" + Path.Combine(root, "app", "preview_entry.py") + "\"";
            start.WorkingDirectory = Path.Combine(root, "app");
            start.UseShellExecute = false;
            start.CreateNoWindow = true;
#if BETA_DISTRIBUTION
            start.EnvironmentVariables["PURIPULY_HEART_BETA_DISTRIBUTION"] = "1";
#else
            start.EnvironmentVariables.Remove("PURIPULY_HEART_BETA_DISTRIBUTION");
#endif
            foreach (string name in Environment.GetEnvironmentVariables().Keys)
            {
                if (name == "PYTHONHOME" || name == "VIRTUAL_ENV" || name == "PYTHONPATH" ||
                    name.StartsWith("PURIPULY_HEART_NATIVE_") || name.StartsWith("FLET_DART_BRIDGE") ||
                    name.EndsWith("_API_KEY") || name.EndsWith("_TOKEN") || name.EndsWith("_PSK"))
                    start.EnvironmentVariables.Remove(name);
            }
            Process.Start(start);
        }
        catch (Exception error)
        {
            MessageBox.Show("Preview startup failed: " + error.GetType().Name, "PuriPuly Preview");
        }
    }
}
