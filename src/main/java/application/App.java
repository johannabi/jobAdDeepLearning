package application;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import data.JobAd;
import preprocessing.Vectorizer;

public class App {

	private static Logger log = LoggerFactory.getLogger(App.class);
	private static String trainingDataPath = "src/main/resources/data/trainingSets/JobAdDB.xlsx";
	private static String studiesPath = "src/main/resources/data/studysubjects.xlsx";
	private static String degreesPath = "src/main/resources/data/degrees.xlsx";
	private static String focusPath = "src/main/resources/data/focuses.xlsx";

	public static void main(String[] args) throws FileNotFoundException {


		Vectorizer vectorizer = new Vectorizer();
		List<JobAd> jobAds = new ArrayList<JobAd>();
		try {
			jobAds = vectorizer.createJobAds(trainingDataPath, studiesPath,
					degreesPath, focusPath);
			for (JobAd jobAd : jobAds) {
				System.out.println(jobAd.getTitle()
						+ "\n Content: " + jobAd.getContent());
				
			}
		} catch (IOException e) {

			e.printStackTrace();
		}
		
		
		
		
		
		
		
		
//		TEST WORD2VEC
//		String filePath = "input.txt";
//
//		log.info("Load & Vectorize Sentences....");
//		// Strip white space before and after for each line
//		SentenceIterator iter = new BasicLineIterator(filePath);
//		
//		// Split on white spaces in the line to get words
//		TokenizerFactory t = new DefaultTokenizerFactory();
//
//		/*
//		 * CommonPreprocessor will apply the following regex to each token:
//		 * [\d\.:,"'\(\)\[\]|/?!;]+ So, effectively all numbers, punctuation symbols and
//		 * some special symbols are stripped off. Additionally it forces lower case for
//		 * all tokens.
//		 */
//		t.setTokenPreProcessor(new CommonPreprocessor());
//
//		log.info("Building model....");
//		Word2Vec vec = new Word2Vec.Builder()
//				.minWordFrequency(5)
//				.iterations(1)
//				.layerSize(100)
//				.seed(42)
//				.windowSize(5)
//				.iterate(iter)
//				.tokenizerFactory(t)
//				.build();
//		log.info("Fitting Word2Vec model....");
//		vec.fit();
//
//		log.info("Writing word vectors to text file....");
//
//		// Prints out the closest 10 words to "day". An example on what to do with these
//		// Word Vectors.
//		log.info("Closest Words:");
//		Collection<String> lst = vec.wordsNearest("day", 10);
//		System.out.println("10 Words closest to 'day': " + lst);
		
		// TODO resolve missing UiServer
//      UiServer server = UiServer.getInstance();
//      System.out.println("Started on port " + server.getPort());

	}

}