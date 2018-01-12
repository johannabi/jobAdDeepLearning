package application;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;

import org.apache.uima.resource.ResourceInitializationException;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/test/java/org/deeplearning4j/bagofwords/vectorizer/TfidfVectorizerTest.java
 * @author Johanna
 *
 */
public class TestApp {

	public static void main(String[] args) throws ResourceInitializationException, FileNotFoundException {

		File rootDir = new ClassPathResource("/output/trainingdata").getFile();
		LabelAwareSentenceIterator iter = new LabelAwareFileSentenceIterator(rootDir);
//		TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
		// RecordReader recRead = new ListStringRecordReader();
		// InputSplit split = new ListStringSplit(data);
		// recRead.initialize(split);

		// DataSetIterator iter = new RecordReaderDataSetIterator(recRead,
		// trainingSet.size(), 11, 12);

		TokenizerFactory tokFac = new UimaTokenizerFactory();

		TfidfVectorizer vectorizer = new TfidfVectorizer.Builder()
				.setMinWordFrequency(1)
				.setStopWords(new ArrayList<String>())
				.setTokenizerFactory(tokFac)
				.setIterator(iter)
				.allowParallelTokenization(false)
				// .labels(labels)
				// .cleanup(true)
				.build();
		
		vectorizer.fit();
		
		//TODO crossvaldation
		vectorizer.transform("text");
		
		
		
		 

	}
}
