package workflow;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.uni_koeln.spinfo.classification.core.data.ClassifyUnit;
import de.uni_koeln.spinfo.classification.core.data.ExperimentConfiguration;
import de.uni_koeln.spinfo.classification.core.helpers.crossvalidation.CrossvalidationGroupBuilder;
import de.uni_koeln.spinfo.classification.core.helpers.crossvalidation.TrainingTestSets;
import de.uni_koeln.spinfo.ml_classification.workflow.FocusJobs;

/**
 * Class contains several methods to handle preprocessing
 * @author Johanna
 *
 */
public class Workflow {

	private static Logger log = LoggerFactory.getLogger(Workflow.class);


	private int numberOfTrainingVectors;
	private int numberOfTestVectors;
	private int numberOfFeatures;
	
	private List<String> featureLabels = new ArrayList<String>();
	private List<String> focusLabels = new ArrayList<String>();
	private List<String> degreeLabels = new ArrayList<String>();
	private List<String> studySubjectLabels = new ArrayList<String>();

	private Set<String> focuses;
	private String stopwordsFilePath = "src/main/resources/data/stopwords.txt";


	public Set<String> getFocuses() {
		return focuses;
	}

	public List<String> getFeatureLabels() {
		return featureLabels;
	}

	public List<String> getFocusLabels() {
		return focusLabels;
	}

	public List<String> getDegreeLabels() {
		return degreeLabels;
	}

	public List<String> getStudySubjectLabels() {
		return studySubjectLabels;
	}

	public int getNumberOfTrainingVectors() {
		return numberOfTrainingVectors;
	}

	public int getNumberOfTestVectors() {
		return numberOfTestVectors;
	}

	public int getNumberOfFeatures() {
		return numberOfFeatures;
	}

	/**
	 * transforms the given trainingdata into vectorized job ads and writes them
	 * into CSVs for crossvalidation. 
	 * @param trainingDataPath path to trainingdata (.xlsx File)
	 * @param focusesPath path to focus labels (.xlsx File)
	 * @param studySubjectsPath path to study subject labels (.xlsx File)
	 * @param degreesPath path to degree labels (.xlsx File)
	 * @param numberOfCrossValidGroups number of crossvalidation folds. for each fold will be produced
	 * one test CSV and one training CSV
	 * @param expConfig contains feature engineering configurations (stemmer, stopword filter,...)
	 * @param currentDir directory where the CSVs will be writen
	 * @throws IOException
	 */
	public void generateTrainingData(String trainingDataPath, String focusesPath, String studySubjectsPath,
			String degreesPath, int numberOfCrossValidGroups, ExperimentConfiguration expConfig, File currentDir)
			throws IOException {
		log.info("Generate Data");
		if (!currentDir.isDirectory())
			currentDir.mkdir();
		FocusJobs jobs = new FocusJobs(stopwordsFilePath);
		List<ClassifyUnit> paragraphs = jobs.getCategorizedAdsFromFile(new File(trainingDataPath), false,
				new File(focusesPath), new File(studySubjectsPath), new File(degreesPath), false);

		paragraphs = jobs.initializeClassifyUnits(paragraphs, true);
		paragraphs = jobs.setFeatures(paragraphs, expConfig.getFeatureConfiguration(), true);
		paragraphs = jobs.setFeatureVectors(paragraphs, expConfig.getFeatureQuantifier(), null);
		
		featureLabels = expConfig.getFeatureQuantifier().getFeatureUnitOrder();
		degreeLabels = new ArrayList<String>(jobs.getDegrees());
		studySubjectLabels = new ArrayList<String>(jobs.getStudySubjects());
		focusLabels = new ArrayList<String>(jobs.getFocuses());
		

		CrossvalidationGroupBuilder<ClassifyUnit> cvgb = new CrossvalidationGroupBuilder<ClassifyUnit>(paragraphs,
				numberOfCrossValidGroups);
		Iterator<TrainingTestSets<ClassifyUnit>> iterator = cvgb.iterator();

		int index = 1;
		while (iterator.hasNext()) {

			TrainingTestSets<ClassifyUnit> testSets = iterator.next();
			List<ClassifyUnit> trainingSet = testSets.getTrainingSet();
			List<ClassifyUnit> testSet = testSets.getTestSet();
			numberOfTrainingVectors = trainingSet.size();
			numberOfTestVectors = testSet.size();
			numberOfFeatures = testSet.get(0).getFeatureVector().length;

			jobs.createCSV("Focus", focusLabels, testSet,
					currentDir.getAbsolutePath() + "/focus/testvectors" + index + ".txt");
			jobs.createCSV("Focus", focusLabels, trainingSet,
					currentDir.getAbsolutePath() + "/focus/trainingvectors" + index + ".txt");
			jobs.createCSV("StudySubject", studySubjectLabels, testSet,
					currentDir.getAbsolutePath() + "/studysubject/testvectors" + index + ".txt");
			jobs.createCSV("StudySubject", studySubjectLabels, trainingSet,
					currentDir.getAbsolutePath() + "/studysubject/trainingvectors" + index + ".txt");
			jobs.createCSV("Degree", degreeLabels, testSet,
					currentDir.getAbsolutePath() + "/degree/testvectors" + index + ".txt");
			jobs.createCSV("Degree", degreeLabels, trainingSet,
					currentDir.getAbsolutePath() + "/degree/trainingvectors" + index + ".txt");
			index++;
			//
		}
		Map<String, Integer> metadata = new HashMap<String, Integer>();
		metadata.put("numberOfTrainingFiles", numberOfTrainingVectors);
		metadata.put("numberOfTestFiles", numberOfTestVectors);
		metadata.put("numberOfFeatures", numberOfFeatures);

		Util.writeMetadata(currentDir, metadata, featureLabels, studySubjectLabels,
				focusLabels, degreeLabels);
	}

	/**
	 * uses methods of dl4j to transform a CSV with multiple labels per row into a MultiDataSet
	 * that can be read by a dl4j neural network
	 * @param filePath path to CSV
	 * @param batchSize number of rows (jobAds) in CSV
	 * @param numberOfLabels number of labels for each jobAd (focus=12,study subject=15,degree=4)
	 * @return dataset that contains the data of CSV
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public MultiDataSet getMultiDataSet(String filePath, int batchSize, int numberOfLabels)
			throws IOException, InterruptedException {
		int numLinesToSkip = 1;
		char delimiter = ',';

		RecordReader rr = new CSVRecordReader(numLinesToSkip, delimiter);
		rr.initialize(new FileSplit(new File(filePath)));

		int inputFrom = 0;
		int inputTo = numberOfFeatures - 1;
		int outputFrom = inputTo + 1;
		int outputTo = inputTo + numberOfLabels;

		MultiDataSetIterator iter = new RecordReaderMultiDataSetIterator.Builder(batchSize).addReader("Reader", rr)
				.addInput("Reader", inputFrom, inputTo).addOutput("Reader", outputFrom, outputTo).build();

		return iter.next();
	}

	/**
	 * reads metadata.txt in the given directory. File contains numbers of
	 * degree, study and focus labels as well as number of features per vector,
	 * number of test vectors and number of training vectors
	 * @param experimentDir directory with CSVs and metadata.txt
	 * @throws IOException
	 */
	public void inizialize(File experimentDir) throws IOException {
		File meta = new File(experimentDir + "/metadata.txt");
		String line = "";
		BufferedReader br = new BufferedReader(new FileReader(meta));
		while ((line = br.readLine()) != null) {
			String parts[] = line.split(": ");
			if (parts[0].equals("numberOfFeatures"))
				numberOfFeatures = Integer.parseInt(parts[1]);
			else if (parts[0].equals("numberOfTestFiles"))
				numberOfTestVectors = Integer.parseInt(parts[1]);
			else if (parts[0].equals("numberOfTrainingFiles"))
				numberOfTrainingVectors = Integer.parseInt(parts[1]);
			else if (parts[0].equals("features"))
				featureLabels = Arrays.asList(parts[1].split("\t"));
			else if (parts[0].equals("studySubject"))
				studySubjectLabels = Arrays.asList(parts[1].split("\t"));
			else if (parts[0].equals("focus"))
				focusLabels = Arrays.asList(parts[1].split("\t"));
			else if (parts[0].equals("degree"))
				degreeLabels = Arrays.asList(parts[1].split("\t"));
		}
		br.close();

	}

}
