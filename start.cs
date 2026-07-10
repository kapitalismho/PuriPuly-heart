using System;
using System.Collections;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Windows.Forms;

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
            MessageBox.Show("python.exe not found in:\n" + Path.Combine(root, "python"),
                "PuriPuly Heart", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }

        if (!File.Exists(mainPy))
        {
            MessageBox.Show("main.py not found in:\n" + Path.Combine(appSrc, "puripuly_heart"),
                "PuriPuly Heart", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }

        Directory.CreateDirectory(dataDir);

        string exeName = Path.GetFileNameWithoutExtension(System.Reflection.Assembly.GetExecutingAssembly().Location);
        bool isDebug = exeName.ToLowerInvariant().Contains("debug");

        Form splash = null;
        if (!isDebug)
        {
            splash = CreateSplash();
            splash.Show();
            Application.DoEvents();
        }

        var psi = new ProcessStartInfo
        {
            FileName = pythonExe,
            UseShellExecute = false,
            CreateNoWindow = !isDebug,
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
                if (splash != null)
                {
                    Thread.Sleep(2000);
                    splash.Close();
                    splash.Dispose();
                }
                proc.WaitForExit();
                return proc.ExitCode;
            }
        }
        catch (Exception ex)
        {
            if (splash != null) { splash.Close(); splash.Dispose(); }
            MessageBox.Show("Failed to start:\n" + ex.Message,
                "PuriPuly Heart", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }

    static Form CreateSplash()
    {
        var form = new Form
        {
            Text = "PuriPuly Heart",
            FormBorderStyle = FormBorderStyle.None,
            StartPosition = FormStartPosition.CenterScreen,
            Size = new Size(320, 160),
            BackColor = Color.FromArgb(30, 30, 30),
            TopMost = true,
            ShowInTaskbar = false,
        };

        var titleLabel = new Label
        {
            Text = "PuriPuly Heart",
            Font = new Font("Segoe UI", 16, FontStyle.Bold),
            ForeColor = Color.FromArgb(240, 128, 180),
            AutoSize = false,
            Size = new Size(320, 40),
            Location = new Point(0, 30),
            TextAlign = ContentAlignment.MiddleCenter,
        };

        var statusLabel = new Label
        {
            Text = "Loading...",
            Font = new Font("Segoe UI", 10),
            ForeColor = Color.FromArgb(180, 180, 180),
            AutoSize = false,
            Size = new Size(320, 25),
            Location = new Point(0, 85),
            TextAlign = ContentAlignment.MiddleCenter,
        };

        form.Controls.Add(titleLabel);
        form.Controls.Add(statusLabel);

        return form;
    }
}
