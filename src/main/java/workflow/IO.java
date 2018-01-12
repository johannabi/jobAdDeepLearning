package workflow;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;

import data.JobAd;
import de.uni_koeln.spinfo.classification.core.data.ClassifyUnit;
import de.uni_koeln.spinfo.ml_classification.data.FocusClassifyUnit;

public class IO {
	
	private String dirPath = "";
	
	public IO(String dirPath) {
		this.dirPath = dirPath;
	}
	
	public void createFoldersForFocuses(Set<String> focuses, 
			List<ClassifyUnit> jobAds, String dirPath) throws IOException {
		this.dirPath = dirPath;
		createFoldersForFocuses(focuses, jobAds);
		
	}
	
	
	public void createFoldersForFocuses(Set<String> focuses,
			List<ClassifyUnit> trainingSet) throws IOException {
		
		File dir = new File(dirPath);
		if(!dir.isDirectory())
			dir.mkdirs();
		
		
		
		//creates folder for each focus
		for (String focus : focuses) {
			File folder = new File(dirPath + "/" + focus);
			folder.mkdirs();
			
		}
		
		for (ClassifyUnit classifyUnit : trainingSet) {
			String content = ((FocusClassifyUnit) classifyUnit).getContent();
			String id = ((FocusClassifyUnit) classifyUnit).getID().toString();
			Map<String, Boolean> currFocuses = ((FocusClassifyUnit) classifyUnit).getInFocus();

			
			for (Map.Entry<String, Boolean> e : currFocuses.entrySet()) {
				if(e.getValue()) {
					String path = dirPath + "/" + e.getKey();
					writeFileToFocus(path, content, id);
				}
			}
		}
		
		
		
	}
	
	public File createCSV(Set<String> labels, List<JobAd> jobAds, String path) throws IOException {
		
		File csv = new File(path);
		if(!csv.exists()) {
			csv.getParentFile().mkdirs();
			csv.createNewFile();
		}
		
		FileWriter fw = new FileWriter(csv);
		for (JobAd jobAd : jobAds) {
			double[] featureValues = jobAd.getFeatureVector();
			Map<String, Boolean> currLabels = jobAd.getInFocus();
			//write feature values
			for (int i = 0; i < featureValues.length; i++) {
				String value = String.valueOf(featureValues[i]);
				fw.write(value + ",");
			}
			//write labels
			StringBuilder sb = new StringBuilder();
			for (String label : labels) {
				if(currLabels.get(label))
					sb.append("1,");
				else
					sb.append("0,");
			}
			String labelString = sb.toString();
			fw.write(labelString.substring(0, labelString.length() - 1) + "\n");
			
		}
		fw.flush();
		fw.close();
		
		return csv;
	}

	private void writeFileToFocus(String path, String content, String id) throws IOException {
		File contentFile = new File(path + "/" + id + ".txt");
		if(!contentFile.exists()) {
			contentFile.getParentFile().mkdirs();
			contentFile.createNewFile();
			
		}
			
//		System.out.println(contentFile.getAbsolutePath());
		FileWriter fw = new FileWriter(contentFile);
		fw.write(content);
		fw.close();
		
	}

}
