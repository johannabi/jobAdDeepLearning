package application;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.uni_koeln.spinfo.classification.core.data.ExperimentConfiguration;
import de.uni_koeln.spinfo.classification.core.data.FeatureUnitConfiguration;
import de.uni_koeln.spinfo.classification.core.featureEngineering.featureWeighting.AbstractFeatureQuantifier;
import de.uni_koeln.spinfo.classification.core.featureEngineering.featureWeighting.LogLikeliHoodFeatureQuantifier;
import de.uni_koeln.spinfo.ml_classification.classifiers.FocusMLKNNClassifier;
import evaluation.Evaluator;
import neuralnet.AbstractComputationGraph;
import neuralnet.AbstractMultiLayerNetwork;
import neuralnet.CNNGenerator;
import neuralnet.DefaultCGGenerator;
import neuralnet.MNISTExample;
import workflow.Workflow;

/**
 * application to evaluate a neural network for a given training dataset by using crossvalidation.
 * The given dataset may appear as .xlsx-File with contents of job ads and their labels. Content will
 * be vectorized by JobAd_IE and feature vectors will be saved in CSVs.
 * If there are already CSVs produced from the given dataset (identical filenames) and with the given
 * feature engineering, these CSVs will be used to feed the neural network.
 * @author Johanna
 *
 */
public class CSVCrossvalidationApp {
	
	private static Logger log = LoggerFactory.getLogger(CSVCrossvalidationApp.class);
	
	///////
	// APP CONFOGURATION
	//////
	private static Workflow wf = new Workflow();
	private static Evaluator eval;
	
	/**number of folds for crossvalidation*/
	private static int crossvalidation = 10;
	
	/** threshold by which probability a label should be set*/
	private static double threshold = 0.8;
	
	/**Map that contains type of label (focus, studysubject, degree) labels for that type*/
	private static Map<String,List<String>> labelsToClassify = new HashMap<String,List<String>>(); 

	/**folder where to put the CSVs for crossvalidation*/
	private static File crossvalidationFolder = new File("src/main/resources/data/trainingSets/crossvalidation");
	
	/**path to job ad excel*/
	private static String trainingDataPath = "src/main/resources/data/trainingSets/JobAdDB.xlsx";
	
	/**path to excel that describes the given study subject labels*/
	private static String studySubjectsPath = "src/main/resources/data/labels/studysubjects.xlsx";
	
	/**path to excel that describes the given focus labels*/
	private static String focusesPath = "src/main/resources/data/labels/focuses.xlsx";
	
	/**path to excel that describes the given degrees*/
	private static String degreesPath = "src/main/resources/data/labels/degrees.xlsx";
	
	private static AbstractComputationGraph graphBuilder = new DefaultCGGenerator();
	private static AbstractMultiLayerNetwork networkBuilder = new MNISTExample();//new CNNGenerator();
	
	////
	// configurations for feature engineering
	////
	static FocusMLKNNClassifier classifier = new FocusMLKNNClassifier(); //nur eingefügt, damit expConfig.toString() keinen null pointer hat
	static boolean ignoreStopwords = true;
	static boolean normalizeInput = true;
	static boolean useStemmer = true;
	static boolean suffixTrees = false;
	static int[] nGrams = {3}; 
	static AbstractFeatureQuantifier quantifier = new LogLikeliHoodFeatureQuantifier();
	///////
	// END
	//////

	public static void main(String[] args) throws IOException, InterruptedException {

		//initialize feature engineering configurations
        FeatureUnitConfiguration fuc = new FeatureUnitConfiguration(normalizeInput, useStemmer, 
    			ignoreStopwords, nGrams, false, 0, suffixTrees);
        ExperimentConfiguration expConfig = new ExperimentConfiguration(fuc, quantifier, 
				classifier, new File(trainingDataPath), null);
        
        //directoryPath contains feature engineering configuration so that you don't need to write the
        //CSVs every time again
        File experimentDir = new File(crossvalidationFolder.getAbsolutePath() + "/" + expConfig.toString());
        
        //if there are no CSVs for this experiment configurations, CSVs are written and put into the
        //experiment directory
        //otherwise metadata for the existing CSVs (e.g. number of features, number of labels,..) will be read
        if(!experimentDir.exists()) 
        	wf.generateTrainingDataCSV(trainingDataPath, focusesPath, studySubjectsPath,
        			degreesPath, crossvalidation, expConfig, experimentDir);
        else
        	 wf.inizialize(experimentDir);
        
        labelsToClassify.put("degree",wf.getDegreeLabels());
		labelsToClassify.put("studysubject",wf.getStudySubjectLabels());
		labelsToClassify.put("focus",wf.getFocusLabels());		
		
        log.info("Initialized project");
        
       
        int batchSizeTrain = wf.getNumberOfTrainingVectors(); 
        int batchSizeTest = wf.getNumberOfTestVectors(); 
        
        //iterates over different label types (degree, studysubject, focus)
        for(Map.Entry<String, List<String>> e : labelsToClassify.entrySet()) {
        	
        	String labelType = e.getKey();
        	
        	Integer numberOfLabels = e.getValue().size();
        	eval = new Evaluator(threshold, e.getValue());
        	
        	//iterates over folds of crossvalidation
        	for (int i = 1; i <= crossvalidation; i++) {
        		
        		//multidataset that contains trainingvectors
            	MultiDataSet trainingData = wf.getMultiDataSet(experimentDir.getAbsolutePath() + "/" + labelType + "/trainingvectors" + i + ".txt", 
            			batchSizeTrain, numberOfLabels);
            	//multidataset that contains testvectors
                MultiDataSet testData = wf.getMultiDataSet(experimentDir.getAbsolutePath() + "/" + labelType + "/testvectors" + i + ".txt",
                		batchSizeTest, numberOfLabels);
          

                //neural net contains an input node for every feature
                int inputNodes = wf.getNumberOfFeatures();
                
                INDArray classified;
                INDArray features;
                INDArray gold;
                
                //generates a neural net with given number of inputs and outputs
                ComputationGraph cgnet = graphBuilder.buildGraph(inputNodes, numberOfLabels);
                MultiLayerNetwork mlnet = networkBuilder.buildGraph(inputNodes, numberOfLabels);
                
//                mlnet.init();
//                log.info(mlnet.toString());
//                mlnet.fit(trainingData);
//                
//                classified = mlnet.output(testData.getFeatures()[0]);
                
                
                cgnet.init();
                //trains the neural network on the given training multidataset (sets weights etc... )
                cgnet.fit(trainingData);
                
                //classifies the given test multidataset in the trained neural network
                //"INDArray classified" contains matrix with probability of each label for each test vector
                classified = cgnet.output(testData.getFeatures())[0];
                
                
                //"INDArray features" contains matrix with feature vectors of each test vector
                features = testData.getFeatures()[0];
                //"INDArray gold" contains matrix with desired labels (0 or 1) for each test vector
                gold = testData.getLabels()[0];
                
                //result of each fold of crossvalidation will be added to evaluator for evaluation
                eval.addResult(labelType, batchSizeTest, features, classified, gold);
                log.info("Finished crossvalidation round: " + i + " for Label " + labelType);
                
            }
        	//evaluates the classification for the given label type
        	 eval.evaluate(labelType, expConfig);
        } 
        
        
       
		
        
     
	}

}
