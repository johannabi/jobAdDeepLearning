package workflow;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.apache.uima.resource.ResourceInitializationException;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import application.App;
import data.JobAd;
import de.uni_koeln.spinfo.classification.core.data.ClassifyUnit;
import de.uni_koeln.spinfo.classification.core.data.ExperimentConfiguration;
import de.uni_koeln.spinfo.classification.core.helpers.crossvalidation.CrossvalidationGroupBuilder;
import de.uni_koeln.spinfo.classification.core.helpers.crossvalidation.TrainingTestSets;
import de.uni_koeln.spinfo.ml_classification.data.FocusClassifyUnit;
import de.uni_koeln.spinfo.ml_classification.workflow.FocusJobs;
import preprocessing.Classifier;
import preprocessing.Vectorizer;

public class Workflow {	

	private FocusJobs jobs;
	private Set<String> focuses;

	public Workflow() {

		try {
			jobs = new FocusJobs("src/main/resources/data/stopwords.txt");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public Set<String> getFocuses() {

		return focuses;
	}

	public void crossvalidate(List<ClassifyUnit> jobAds) throws ResourceInitializationException,
	IOException {
		int folds = 10;
		String trainingFolder = "src/main/resources/labeled";
		String testFolder = "src/main/resources/unlabeled";
		CrossvalidationGroupBuilder<ClassifyUnit> cvgb = new CrossvalidationGroupBuilder<ClassifyUnit>(jobAds, folds);
		Iterator<TrainingTestSets<ClassifyUnit>> iterator = cvgb.iterator();
		List<JobAd> testedJobAds = new ArrayList<JobAd>();

		int it = 1;
		while (iterator.hasNext()) {
			String currentTrainingFolder = trainingFolder + it + "/";
			String currentTestFolder = testFolder + it + "/";
			TrainingTestSets<ClassifyUnit> testSets = iterator.next();

			List<ClassifyUnit> trainingSet = testSets.getTrainingSet();
			IO io = new IO(currentTrainingFolder);
			io.createFoldersForFocuses(focuses, trainingSet, currentTrainingFolder);			
			
			List<ClassifyUnit> testSet = testSets.getTestSet();
			io.createFoldersForFocuses(focuses, testSet, currentTestFolder);
			Vectorizer vectorizer = new Vectorizer();
			List<JobAd> vectorizedTestSet = vectorizer.vectorize(currentTrainingFolder, testSet);
//			System.out.println(vectorizedTestSet.size() + "!");
			
			//TODO classify vectors
			List<JobAd> classifiedTestSet = new ArrayList<JobAd>();
			Classifier classifier = new Classifier();
			classifier.classify(vectorizedTestSet, vectorizer.getVectorizer());
			
			testedJobAds.addAll(classifiedTestSet);
			it++;
		}
	}
	
	public void rrCrossvalidate (List<ClassifyUnit> jobAds) {
		int folds = 10;
		CrossvalidationGroupBuilder<ClassifyUnit> cvgb = new CrossvalidationGroupBuilder<ClassifyUnit>(jobAds, folds);
		Iterator<TrainingTestSets<ClassifyUnit>> iterator = cvgb.iterator();
		List<JobAd> testedJobAds = new ArrayList<JobAd>();
		

		int it = 1;
		while (iterator.hasNext()) {
			TrainingTestSets<ClassifyUnit> testSets = iterator.next();
			List<ClassifyUnit> trainingSet = testSets.getTrainingSet();			
			List<ClassifyUnit> testSet = testSets.getTestSet();
			
			RecordReader rr = new ListStringRecordReader();
			//TODO input für Record Reader implementieren
		}
	}

	public List<JobAd> createJobAds(String trainingDataPath, String studiesPath, String degreesPath, String focusPath)
			throws IOException {

		File trainingData = new File(trainingDataPath);
		File studiesFile = new File(studiesPath);
		File degreesFile = new File(degreesPath);
		File focusesFile = new File(focusPath);

		List<ClassifyUnit> classifyUnits = jobs.getCategorizedAdsFromFile(trainingData, false, focusesFile, studiesFile,
				degreesFile, false);

		focuses = jobs.getFocuses();

		List<JobAd> jobAds = new ArrayList<JobAd>();

		for (ClassifyUnit cu : classifyUnits) {
			FocusClassifyUnit fcu = (FocusClassifyUnit) cu;

			JobAd jobAd = new JobAd(fcu.getTitle(), cu.getContent(), cu.getID(), fcu.getContentHTML());
			jobAd.setInFocus(fcu.getInFocus());
			jobAd.setDegrees(fcu.getDegrees());
			jobAd.setStudySubjects(fcu.getStudySubjects());

			// TODO vectorize content

			jobAds.add(jobAd);
		}

		return jobAds;

	}
	
	public List<JobAd> createVectors(List<JobAd> jobAds){


		
		
		return jobAds;
	}

}
