package workflow;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.PosUimaTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
//import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.uni_koeln.spinfo.classification.core.data.ClassifyUnit;
import de.uni_koeln.spinfo.ml_classification.data.FocusClassifyUnit;

public class Preprocessor {
	
	private static Logger log = LoggerFactory.getLogger(Preprocessor.class);
	private int numberOfFeatures = 0;
	private int numberOfLabels = 0;

	public INDArray vectorizeJobAds(List<ClassifyUnit> jobAds) {
		
		TokenizerFactory factory = new DefaultTokenizerFactory();
		List<String> allowedTags = new ArrayList<String>();
		allowedTags.add("NN");
		//ngram = 3 - 4 tokens
		TokenizerFactory tf = new PosUimaTokenizerFactory(allowedTags);
//		TokenizerFactory tf = new NGramTokenizerFactory(factory, 3, 4);
//		List<String> allTokens = new ArrayList<String>();
		Set<String> tokenSet = new HashSet<String>();
		
		
		//TODO tokenize, weight, vectorize,...
		for (ClassifyUnit cu : jobAds) {
			String content = cu.getContent();
			Tokenizer tokenized = tf.create(content);
			List<String> currentTokens = tokenized.getTokens();
			for (String string : currentTokens) {
				log.info(string);
			}
			
			tokenSet.addAll(currentTokens);
		}
		
		log.info(tokenSet.size() + "");
		
		return null;
	}
	
	public MultiDataSet getMultiDataSet(List<ClassifyUnit> jobAds, List<String> labels, String labelType) {
		
		numberOfFeatures= jobAds.get(0).getFeatureVector().length;
		numberOfLabels = labels.size();
		int rows = jobAds.size();
		

		double featureMatrix[][] = new double[rows][numberOfFeatures];
		double labelMatrix[][] = new double[rows][numberOfLabels];
		
		for (int i = 0; i < featureMatrix.length; i++) {
			FocusClassifyUnit fcu = (FocusClassifyUnit) jobAds.get(i);
			double values[] = fcu.getFeatureVector();
			
			//add feature values
			for (int j = 0; j < values.length; j++) {
				featureMatrix[i][j] = values[j];
			}
			
			//add labels
			Map<String, Boolean> labelMap = new HashMap<String, Boolean>();
			if(labelType.equals("StudySubject"))
				labelMap = fcu.getStudySubjects();
			if(labelType.equals("Degree"))
				labelMap = fcu.getDegrees();
			if(labelType.equals("Focus"))
				labelMap = fcu.getInFocus();
			for(int j = 0; j < numberOfLabels; j++) { //iteriert über label-array
				
				
				
				if(labelMap.get(labels.get(j)))
					labelMatrix[i][j] = 1;
				else
					labelMatrix[i][j] = 0;
			}
		}
		
//		for (int i = 0; i < matrix.length; i++) {
//			for (int j = 0; j < matrix[i].length; j++) {
//				System.out.println(matrix[i][j]);
//			}
//		}
//		log.info("FeatureVector Size: " + jobAds.get(0).getFeatureVector().length);
//		log.info("Labels: " + studyLabels.size());
//		log.info("JobAds: " + jobAds.size());
//		log.info("Matrix Rows: " + matrix.length);
//		log.info("Matrix Columns: " + matrix[0].length);
		INDArray featureArray = Nd4j.create(featureMatrix);
		INDArray labelArray = Nd4j.create(labelMatrix);
		MultiDataSet set = new MultiDataSet(featureArray,labelArray);
		
		return set;
	}

	public int getNumberOfFeatures() {
		return numberOfFeatures;
	}

	public int getNumberOfLabels() {
		return numberOfLabels;
	}

}
