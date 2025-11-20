using System.IO;
using Newtonsoft.Json;

namespace TIA_Openness_Demo
{
    public static class JsonExporter
    {
        public static void Save(string exportDir, string filename, object data)
        {
            Directory.CreateDirectory(exportDir);

            string json = JsonConvert.SerializeObject(data, Formatting.Indented);
            File.WriteAllText(Path.Combine(exportDir, filename), json);
        }
    }
}
