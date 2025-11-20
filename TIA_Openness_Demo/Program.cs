using System;
using System.IO;
using Siemens.Engineering;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;

namespace TIA_Openness_Demo
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== TIA Openness Demo ===");

            string projectPath = @"C:\Users\hbui11\Documents\Automation\Project1\Project1.ap17";
            string exportDir = Path.Combine(Environment.CurrentDirectory, "Export");

            Directory.CreateDirectory(exportDir);

            using (TiaPortal tia = new TiaPortal(TiaPortalMode.WithoutUserInterface))
            {
                Console.WriteLine("[+] TIA Portal started.");

                Project project = TiaHelper.OpenProject(tia, projectPath);
                Console.WriteLine($"[+] Loaded project: {projectPath}");

                // Export logic to XML using available API
                try
                {
                    XmlExporter.ExportProjectLogic(project, exportDir);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[ERROR] Export failed: {ex.Message}");
                    Console.WriteLine($"Stack trace: {ex.StackTrace}");
                }
                
                Console.WriteLine("[+] Export completed.");
            }

            Console.WriteLine("[+] Done.");
        }
    }
}
