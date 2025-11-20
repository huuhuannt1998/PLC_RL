using System.Collections.Generic;
using System.IO;
using Siemens.Engineering;

namespace TIA_Openness_Demo
{
    public static class TiaHelper
    {
        public static Project OpenProject(TiaPortal tia, string projectPath)
        {
            return tia.Projects.Open(new FileInfo(projectPath));
        }

        // NOTE: The following methods require Siemens.Engineering.HW.dll and Siemens.Engineering.SW.dll
        // which are not available in this TIA Portal installation.
        // Uncomment when the required assemblies are available.

        /*
        public static List<Device> ListDevices(Project project)
        {
            var list = new List<Device>();

            foreach (var device in project.Devices)
                list.Add(device);

            return list;
        }

        public static List<SoftwareContainer> ListBlocks(Project project)
        {
            var list = new List<SoftwareContainer>();

            foreach (var device in project.Devices)
            {
                foreach (var item in device.DeviceItems)
                {
                    if (item.SoftwareContainer != null)
                        list.Add(item.SoftwareContainer);
                }
            }

            return list;
        }
        */
    }
}
