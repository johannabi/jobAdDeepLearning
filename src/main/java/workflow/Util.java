package workflow;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import de.uni_koeln.spinfo.classification.zoneAnalysis.data.CategoryResult;
import de.uni_koeln.spinfo.ml_classification.data.MLCategoryResult;
import de.uni_koeln.spinfo.ml_classification.data.MLExperimentResult;

public class Util {

	public static void writeMetadata(File dir, Map<String, Integer> metadata, List<String> featureLabels,
			List<String> studySubjectLabels, List<String> focusLabels, List<String> degreeLabels) throws IOException {

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

	public static void exportResult(Map<String, List<MLExperimentResult>> results) throws IOException {
		String path = "result.xlsx";
		File export = new File(path);
		if (!export.exists())
			export.createNewFile();

		XSSFWorkbook wb = new XSSFWorkbook();

		for (Map.Entry<String, List<MLExperimentResult>> e : results.entrySet()) {
			List<MLExperimentResult> labelResults = e.getValue();
			MLExperimentResult currResult = labelResults.get(0);
			List<MLCategoryResult> catResults;
			

			List<String> headRow = new ArrayList<String>();
			headRow.add("learningRate");
			headRow.add("threshold");
			headRow.add("backp");
			headRow.add("layer1");
			headRow.add("nIn1");
			headRow.add("nOut1");
			headRow.add("act1");
			headRow.add("weight1");
			headRow.add("layer2");
			headRow.add("nIn2");
			headRow.add("nOut2");
			headRow.add("act2");
			headRow.add("weight2");
			headRow.add("layer3");
			headRow.add("nIn3");
			headRow.add("nOut3");
			headRow.add("act3");
			headRow.add("weight3");
			headRow.add("layer4");
			headRow.add("nIn4");
			headRow.add("nOut4");
			headRow.add("act4");
			headRow.add("weight4");
			int startResult = headRow.size();
			headRow.add("hammingLoss");
			
			Map<String, Double> macro = currResult.getMacroAveraging();
			for (Map.Entry<String, Double> m : macro.entrySet()) {
				headRow.add(m.getKey());
			}
			
			Map<String,Double> micro = currResult.getMicroAveraging();
			for (Map.Entry<String, Double> m : micro.entrySet()) {
				headRow.add(m.getKey());
			}
			
			catResults = currResult.getMLCategoryEvaluations();
			for (CategoryResult cat : catResults) {
				MLCategoryResult mlCat = (MLCategoryResult) cat;
				headRow.add(mlCat.getLabel() + " P");
				headRow.add(mlCat.getLabel() + " R");
				headRow.add(mlCat.getLabel() + " F");
			}
			

			XSSFSheet sheet = wb.createSheet(e.getKey());
			int r = 0;

			// headrow
			Row row = sheet.createRow(r++);
			Cell cell = null;
			for (int i = 0; i < headRow.size(); i++) {
				cell = row.createCell(i);
				cell.setCellValue(headRow.get(i));
			}

			// results
			for (MLExperimentResult result : labelResults) {
				row = sheet.createRow(r++);

				//config
				String[] expConfig = result.getExperimentConfiguration().split("_layer=");
				String[] netConfig = expConfig[0].split("_");
				
				// net config
				for (int i = 0; i < netConfig.length; i++) {
					String currConfig = netConfig[i];
//					System.out.println(currConfig);
					if (currConfig.contains("learningrate")) {
						cell = row.createCell(0);
						cell.setCellValue(currConfig.replaceAll("learningrate=", ""));
					} else if (currConfig.contains("thres")) {
						cell = row.createCell(1);
						cell.setCellValue(currConfig.replaceAll("thres=", ""));
					} else if (currConfig.contains("backpr")) {
						cell = row.createCell(2);
						cell.setCellValue(currConfig.replaceAll("backpr=", ""));
						
					} else {
						continue;
					}
				}
				int c = 3;
				//iterates over layer configs
				for (int i = 1; i < expConfig.length; i++) {
					String[] layerConfig = expConfig[i].split("_");
					
					cell = row.createCell(c++);
					cell.setCellValue(layerConfig[0]); //layer name
					
					cell = row.createCell(c++);
					cell.setCellValue(layerConfig[1].replaceAll("nIn=", ""));
					
					cell = row.createCell(c++);
					cell.setCellValue(layerConfig[2].replaceAll("nOut=", ""));
					
					cell = row.createCell(c++);
					cell.setCellValue(layerConfig[3].replaceAll("act=", ""));
					
					cell = row.createCell(c++);
					cell.setCellValue(layerConfig[4].replaceAll("w=", ""));
	
				}
				
				c = startResult;

				//evaluation result
				cell = row.createCell(c++);
				cell.setCellValue(result.getHammingLoss());

				int i = 0;
				// macro / micro
				for(Map.Entry<String, Double> m : result.getMacroAveraging().entrySet()) {
					i = c;
					if(m.getKey().equals(headRow.get(i++))) {
						cell = row.createCell(i-1);
						cell.setCellValue(m.getValue());
					} else if(m.getKey().equals(headRow.get(i++))) {
						cell = row.createCell(i-1);
						cell.setCellValue(m.getValue());
					} else if (m.getKey().equals(headRow.get(i++))) {
						cell = row.createCell(i-1);
						cell.setCellValue(m.getValue());
					}
				}
				c = i;
				for(Map.Entry<String, Double> m : result.getMicroAveraging().entrySet()) {
					i = c;
					if(m.getKey().equals(headRow.get(i++))) {
						cell = row.createCell(i-1);
						cell.setCellValue(m.getValue());
					} else if(m.getKey().equals(headRow.get(i++))) {
						cell = row.createCell(i-1);
						cell.setCellValue(m.getValue());
					} else if (m.getKey().equals(headRow.get(i++))) {
						cell = row.createCell(i-1);
						cell.setCellValue(m.getValue());
					}
				}
				c = c + 3;
				//cat results
				catResults = result.getMLCategoryEvaluations();	
				for (MLCategoryResult cat : catResults) {

					cell = row.createCell(c++);
					cell.setCellValue(cat.getPrecision());
					
					cell = row.createCell(c++);
					cell.setCellValue(cat.getRecall());
					
					cell = row.createCell(c++);
					cell.setCellValue(cat.getF1Score());
				}
				
			}

		}

		FileOutputStream fos = new FileOutputStream(path);
		wb.write(fos);
		wb.close();

	}

}
