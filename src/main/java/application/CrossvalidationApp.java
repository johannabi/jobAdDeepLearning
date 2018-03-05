package application;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import data.DLExperimentConfiguration;
import de.uni_koeln.spinfo.classification.core.data.FeatureUnitConfiguration;
import de.uni_koeln.spinfo.classification.core.featureEngineering.featureWeighting.AbstractFeatureQuantifier;
import de.uni_koeln.spinfo.classification.core.featureEngineering.featureWeighting.LogLikeliHoodFeatureQuantifier;
import de.uni_koeln.spinfo.ml_classification.data.MLExperimentResult;
import workflow.Util;
import workflow.Workflow;

/**
 * App to crossvalidate the configuration of a neural network
 * @author Johanna
 *
 */
public class CrossvalidationApp {

	///////
	// APP CONFOGURATION
	//////
	private static Workflow wf = new Workflow();
	/** number of folds for crossvalidation */
	private static int crossvalidation = 10;

//	/** threshold by which probability a label should be set */
//	private static double threshold = 0.8;

	/** path to job ad excel */
	private static String trainingDataPath = "src/main/resources/data/trainingSets/JobAdDB.xlsx";

	/** path to excel that describes the given study subject labels */
	private static String studySubjectsPath = "src/main/resources/data/labels/studysubjects.xlsx";

	/** path to excel that describes the given focus labels */
	private static String focusesPath = "src/main/resources/data/labels/focuses.xlsx";

	/** path to excel that describes the given degrees */
	private static String degreesPath = "src/main/resources/data/labels/degrees.xlsx";

	////
	// configurations for feature engineering
	////
	static boolean ignoreStopwords = true;
	static boolean normalizeInput = true;
	static boolean useStemmer = true;
	static boolean suffixTrees = false;
	static int[] nGrams = { 3 };
	static AbstractFeatureQuantifier quantifier = new LogLikeliHoodFeatureQuantifier();
	
	///////
	// END
	//////

	public static void main(String[] args) throws IOException {

		List<Integer> firstHiddenNodes = new ArrayList<Integer>();
		List<Integer> secondHiddenNodes = new ArrayList<Integer>();
		List<Double> learningRate = new ArrayList<Double>();
		List<Double> threshold = new ArrayList<Double>();
		List<String[]> activation = new ArrayList<String[]>();
		
		activation.add(new String[] {"relu", "relu", "softmax"});
		activation.add(new String[] {"sigmoid", "relu", "relu"});
		activation.add(new String[] {"relu", "relu", "relu"});
		
//		threshold.add(0.4);
		threshold.add(0.5);
//		threshold.add(0.7);
//		threshold.add(0.8);		
		
		firstHiddenNodes.add(500);
		firstHiddenNodes.add(250);
//		firstHiddenNodes.add(150);
		
//		secondHiddenNodes.add(25);
//		secondHiddenNodes.add(50);
		secondHiddenNodes.add(500);
		secondHiddenNodes.add(100);
		
//		learningRate.add(0.0001);
		learningRate.add(0.001);
		learningRate.add(0.01);
//		learningRate.add(0.1);

		// initialize feature engineering configurations
		FeatureUnitConfiguration fuc = new FeatureUnitConfiguration(normalizeInput, useStemmer, ignoreStopwords, nGrams,
				false, 0, suffixTrees);
		DLExperimentConfiguration expConfig = new DLExperimentConfiguration(fuc, quantifier,
				new File(trainingDataPath));

		Map<String, List<MLExperimentResult>> results = wf.crossvalidate(trainingDataPath, focusesPath, studySubjectsPath, degreesPath, expConfig, crossvalidation,
				firstHiddenNodes, secondHiddenNodes, learningRate, threshold, activation);
		Util.exportResult(results);
	}

}
