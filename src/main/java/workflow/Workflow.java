package workflow;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.apache.uima.resource.ResourceInitializationException;
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

	public void crossvalidate(List<ClassifyUnit> jobAds) throws ResourceInitializationException {
		int folds = 10;
		CrossvalidationGroupBuilder<ClassifyUnit> cvgb = new CrossvalidationGroupBuilder<ClassifyUnit>(jobAds, folds);
		Iterator<TrainingTestSets<ClassifyUnit>> iterator = cvgb.iterator();
		Vectorizer vectorizer = new Vectorizer();

		while (iterator.hasNext()) {
			TrainingTestSets<ClassifyUnit> testSets = iterator.next();

			List<ClassifyUnit> trainingSet = testSets.getTrainingSet();
			List<ClassifyUnit> testSet = testSets.getTestSet();
			
			vectorizer.vectorize(trainingSet);
			
			
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
