using System;
using System.IO;
using Siemens.Engineering;
using System.Reflection;

namespace TIA_Openness_Demo
{
    public static class XmlExporter
    {
        public static void ExportProjectLogic(Project project, string exportDir)
        {
            Directory.CreateDirectory(exportDir);
            int exportCount = 0;

            Console.WriteLine("[+] Starting logic export...");

            try
            {
                // Get devices collection using reflection
                var devicesProperty = project.GetType().GetProperty("Devices");
                if (devicesProperty == null)
                {
                    Console.WriteLine("[ERROR] Could not access Devices property");
                    return;
                }

                var devices = devicesProperty.GetValue(project);
                var devicesEnumerable = devices as System.Collections.IEnumerable;
                
                if (devicesEnumerable == null)
                {
                    Console.WriteLine("[ERROR] Devices is not enumerable");
                    return;
                }

                foreach (var device in devicesEnumerable)
                {
                    var deviceName = device.GetType().GetProperty("Name")?.GetValue(device)?.ToString() ?? "UnknownDevice";
                    Console.WriteLine($"[+] Processing device: {deviceName}");

                    // Get DeviceItems
                    var deviceItemsProperty = device.GetType().GetProperty("DeviceItems");
                    if (deviceItemsProperty == null) continue;

                    var deviceItems = deviceItemsProperty.GetValue(device) as System.Collections.IEnumerable;
                    if (deviceItems == null) continue;

                    foreach (var item in deviceItems)
                    {
                        // Try to get SoftwareContainer
                        var softwareContainerProperty = item.GetType().GetProperty("SoftwareContainer");
                        if (softwareContainerProperty == null) continue;

                        var softwareContainer = softwareContainerProperty.GetValue(item);
                        if (softwareContainer == null) continue;

                        // Get Software (PlcSoftware)
                        var softwareProperty = softwareContainer.GetType().GetProperty("Software");
                        if (softwareProperty == null) continue;

                        var software = softwareProperty.GetValue(softwareContainer);
                        if (software == null) continue;

                        var softwareType = software.GetType().Name;
                        Console.WriteLine($"  [+] Found software: {softwareType}");

                        // Get BlockGroup
                        var blockGroupProperty = software.GetType().GetProperty("BlockGroup");
                        if (blockGroupProperty == null)
                        {
                            Console.WriteLine($"  [!] No BlockGroup property found");
                            continue;
                        }

                        var blockGroup = blockGroupProperty.GetValue(software);
                        if (blockGroup == null) continue;

                        // Get Blocks collection
                        var blocksProperty = blockGroup.GetType().GetProperty("Blocks");
                        if (blocksProperty == null) continue;

                        var blocks = blocksProperty.GetValue(blockGroup) as System.Collections.IEnumerable;
                        if (blocks == null) continue;

                        // Export each block
                        foreach (var block in blocks)
                        {
                            try
                            {
                                var blockName = block.GetType().GetProperty("Name")?.GetValue(block)?.ToString() ?? "UnknownBlock";
                                
                                // Call Export method
                                var exportMethod = block.GetType().GetMethod("Export", new[] { typeof(FileInfo), Type.GetType("Siemens.Engineering.ExportOptions") });
                                if (exportMethod == null)
                                {
                                    Console.WriteLine($"    [!] Export method not found for block: {blockName}");
                                    continue;
                                }

                                string fileName = Path.Combine(exportDir, $"{deviceName}_{blockName}.xml");
                                var fileInfo = new FileInfo(fileName);

                                // Get ExportOptions.WithDefaults
                                var exportOptionsType = Type.GetType("Siemens.Engineering.ExportOptions, Siemens.Engineering");
                                if (exportOptionsType != null)
                                {
                                    var withDefaultsProperty = exportOptionsType.GetProperty("WithDefaults", BindingFlags.Public | BindingFlags.Static);
                                    if (withDefaultsProperty != null)
                                    {
                                        var exportOptions = withDefaultsProperty.GetValue(null);
                                        exportMethod.Invoke(block, new object[] { fileInfo, exportOptions });
                                        Console.WriteLine($"    [✓] Exported: {fileName}");
                                        exportCount++;
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"    [ERROR] Failed to export block: {ex.Message}");
                            }
                        }
                    }
                }

                Console.WriteLine($"[+] Successfully exported {exportCount} blocks to XML");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] Export failed: {ex.Message}");
                throw;
            }
        }
    }
}
