package evaluation;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.uni_koeln.spinfo.classification.core.data.ClassifyUnit;
import de.uni_koeln.spinfo.classification.core.data.ExperimentConfiguration;
import de.uni_koeln.spinfo.classification.zoneAnalysis.data.CategoryResult;
import de.uni_koeln.spinfo.ml_classification.data.FocusClassifyUnit;
import de.uni_koeln.spinfo.ml_classification.data.MLCategoryResult;
import de.uni_koeln.spinfo.ml_classification.data.MLExperimentResult;
import de.uni_koeln.spinfo.ml_classification.evaluation.MLEvaluator;

/**
 * Class contains methods to transform the result of dl4j neural network into datastructure
 * that can be understood by methods of JobAd_IE
 * @author Johanna
 *
 */
public class Evaluator {
	
	private static Logger log = LoggerFactory.getLogger(Evaluator.class);

	private Map<ClassifyUnit, Map<String, Boolean>> cuClassified = new HashMap<ClassifyUnit, Map<String, Boolean>>();
	private double threshold;
	private List<String> labels;
	
	
	public Evaluator(double threshold, List<String> labels) {
		this.threshold = threshold;
		this.labels = labels;
	}
	
	public Evaluator(List<String> labels) {
		this.labels = labels;
		this.threshold = 0.5;
	}

	/**
	 * adds the result of classification to the Evaluator. after the result of each crossvalidation step is added
	 * the Evaluator can evaluate all results.
	 * @param label label type that had been classified
	 * @param batchSizeTest number of test vectors
	 * @param features features of the test vector
	 * @param classified classified probability for each test vector and each label
	 * @param gold desired labels (1 or 0) for each test vector
	 */
	public void addResult(String label, int batchSizeTest, INDArray features, INDArray classified, INDArray gold) {
		
		// iterate over testData
		for (int i = 0; i < features.size(0); i++) { //features.size(0) corresponds to the number of rows "features"
			FocusClassifyUnit fcu = new FocusClassifyUnit(""); // TODO content

			INDArray currentFeatures = features.getRow(i);
			INDArray currentClassified = classified.getRow(i);
			INDArray currentGold = gold.getRow(i);
	
			Map<String, Boolean> classifiedMap = new HashMap<String, Boolean>();
			Map<String, Boolean> goldMap = new HashMap<String, Boolean>();
			Map<String, Double> rankedMap = new HashMap<String, Double>();

			// create classifiedMap (dichotom and ranking)
			for (int j = 0; j < classified.size(1); j++) { //classified.size(1) corresponds to the number of columns of "classified"
				double probability = currentClassified.getDouble(j);
//				log.info(probability + "");
				rankedMap.put(labels.get(j), probability);
				if (probability > threshold)
					classifiedMap.put(labels.get(j), true); // TODO label
				else
					classifiedMap.put(labels.get(j), false);
			}

			// create goldMap
			for (int j = 0; j < gold.size(1); j++) {
				double labelValue = currentGold.getDouble(j);
				if (labelValue == 1.0)
					goldMap.put(labels.get(j), true); // TODO label
				else
					goldMap.put(labels.get(j), false);
			}
			
			if (label.equals("Focus")) { 
				fcu.setInFocus(goldMap);
				fcu.setRanking(rankedMap);
				cuClassified.put(fcu, classifiedMap);
			} else if (label.equals("Degree")) {
				fcu.setDegrees(goldMap);
				fcu.setExtractedDegrees(classifiedMap);
				cuClassified.put(fcu, null);
			}
			else if (label.equals("StudySubject")) {
				fcu.setStudySubjects(goldMap);
				fcu.setExtractedStudies(classifiedMap);
				cuClassified.put(fcu, null);

			}


			
		}

	}

	/**
	 * evaluates the collected results for one label type with methods of JobAd_IE. 
	 * The evaluation result will be printed in console
	 * @param labelType
	 * @param expConfig
	 */
	public Map<String, MLExperimentResult> evaluate(String labelType, ExperimentConfiguration expConfig) {

		MLEvaluator evaluator = new MLEvaluator();
		Map<String, MLExperimentResult> results = null;
		List<String> categories = null;
		
		if(labelType.equals("StudySubject")) {
			results = evaluator.evaluateStudySubjects(cuClassified, expConfig, categories, labels);
		} else if(labelType.equals("Degree")) {
			results = evaluator.evaluateDegrees(cuClassified, expConfig, categories, labels);
		} else if(labelType.equals("Focus")) {
			results = evaluator.evaluateFocuses(cuClassified, expConfig, categories, labels);
		}

		for (Map.Entry<String, MLExperimentResult> e : results.entrySet()) {
			System.out.println("++++" + e.getKey() + "++++");
			MLExperimentResult result = e.getValue();
			List<MLCategoryResult> catResults = result.getMLCategoryEvaluations();
			for (CategoryResult cr : catResults) {
				System.out.println(
						"TP: " + cr.getTP() + " - FP: " + cr.getFP() + " - FN: " + cr.getFN() + " - TN: " + cr.getTN());
				System.out.println(cr);
			}
			System.out.println(result.getMacroAveraging());
			System.out.println(result.getMicroAveraging());

			System.out.println("Hamming Loss: \t" + result.getHammingLoss());
			System.out.println("One Error: \t" + result.getOneError());
			System.out.println("Coverage: \t" + result.getCoverage());

			System.out.println("Average Precision: " + result.getAverPrec());
			System.out.println("Precision: " + result.getPrecision());
			System.out.println("Average Recall: " + result.getAverRec());
			System.out.println("Recall: " + result.getRecall());
			System.out.println("F-Measure: " + result.getF1Measure());
			System.out.println("Average F-Measure: " + result.getAverF1());
			System.out.println("Accuracy: " + result.getAccuracy());
			System.out.println("Classification Accuracy: " + result.getClassificationAccuracy());
			System.out.println("----------------------------------------------------------------------");

		}
		
		return results;
	}

}
