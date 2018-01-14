package workflow;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Map;



public class Util {
	
	public static void writeMetadata(File dir, Map<String, Integer> metadata, List<String> featureLabels, List<String> studySubjectLabels, 
			List<String> focusLabels, List<String> degreeLabels) throws IOException {
		
		File meta = new File(dir.getAbsolutePath() + "/metadata.txt");
		FileWriter fw = new FileWriter(meta);
		for (Map.Entry<String, Integer> e : metadata.entrySet()) {
			fw.write(e.getKey() + ": " + e.getValue() + "\n");
		}
		fw.write("features: ");
		for (String string : featureLabels) {
			fw.write(string + "\t");
		}
		fw.write("\n");
		fw.write("studySubject: ");
		for (String string : studySubjectLabels) {
			fw.write(string + "\t");
		}
		fw.write("\n");
		fw.write("focus: ");
		for (String string : focusLabels) {
			fw.write(string + "\t");
		}
		fw.write("\n");
		fw.write("degree: ");
		for (String string : degreeLabels) {
			fw.write(string + "\t");
		}
		
		fw.close();
		
		
	}

}
