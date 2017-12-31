package preprocessing;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import application.App;
import data.JobAd;
import de.uni_koeln.spinfo.classification.core.data.ClassifyUnit;
import de.uni_koeln.spinfo.ml_classification.data.FocusClassifyUnit;
import de.uni_koeln.spinfo.ml_classification.workflow.FocusJobs;

public class Vectorizer {
	
	private static Logger log = LoggerFactory.getLogger(App.class);
	
	
	private FocusJobs jobs;
	
	
	public Vectorizer() {
		
		try {
			jobs = new FocusJobs("src/main/resources/data/stopwords.txt");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public List<JobAd> createJobAds(String trainingDataPath, String studiesPath,
			String degreesPath, String focusPath) throws IOException {
		
		File trainingData = new File(trainingDataPath);
		File studiesFile = new File(studiesPath);
		File degreesFile = new File(degreesPath);
		File focusesFile = new File(focusPath);
		
		List<ClassifyUnit> classifyUnits = jobs.getCategorizedAdsFromFile(trainingData, false, focusesFile,
				studiesFile, degreesFile, false);
		
		List<JobAd> jobAds = new ArrayList<JobAd>();
		
		for (ClassifyUnit cu : classifyUnits) {
			JobAd jobAd = new JobAd(((FocusClassifyUnit) cu).getTitle(), cu.getContent(),
					cu.getID(), ((FocusClassifyUnit) cu).getContentHTML());
			
			//TODO vectorize content
			
			
			
			jobAds.add(jobAd);
		}
		
		return jobAds;
		
	}
	
	
	

}
