package preprocessing;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import org.apache.uima.resource.ResourceInitializationException;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.ListStringSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.bagofwords.vectorizer.BaseTextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import application.App;
import data.JobAd;
import de.uni_koeln.spinfo.classification.core.data.ClassifyUnit;
import de.uni_koeln.spinfo.ml_classification.data.FocusClassifyUnit;

public class Vectorizer {

	private static Logger log = LoggerFactory.getLogger(App.class);
	private TfidfVectorizer vectorizer;
	
	public TfidfVectorizer getVectorizer() {
		return vectorizer;
	}

	public void initialize(List<ClassifyUnit> trainingSet) throws IOException, InterruptedException {

		List<List<String>> data = new ArrayList<List<String>>();
		// construct data object
		for (ClassifyUnit cu : trainingSet) {
			String content = cu.getContent();
		}

		RecordReader recRead = new ListStringRecordReader();
		InputSplit split = new ListStringSplit(data);
		recRead.initialize(split);

		DataSetIterator iter = new RecordReaderDataSetIterator(recRead, trainingSet.size(), 11, 12);

	}
	
	

	public List<JobAd> vectorize(String labeledFolder, List<ClassifyUnit> toVectorize)
			throws ResourceInitializationException, FileNotFoundException {

		// build vectorizer
		File rootDir = new File(labeledFolder);
		LabelAwareSentenceIterator iter = new LabelAwareFileSentenceIterator(rootDir);

		TokenizerFactory tokFac = new UimaTokenizerFactory();

		vectorizer = new TfidfVectorizer.Builder()
				.setMinWordFrequency(1)
				.setStopWords(new ArrayList<String>())
				.setTokenizerFactory(tokFac)
				.setIterator(iter)
				.allowParallelTokenization(false)
				// .labels(labels)
				// .cleanup(true)
				.build();

		vectorizer.fit();
//		System.out.println(vectorizer.getIndex());
		
		List<JobAd> toReturn = new ArrayList<JobAd>();
		// vectorize job ads
		for (ClassifyUnit cu : toVectorize) {
			JobAd jobAd = (JobAd) cu;
			String content = jobAd.getContent();
			INDArray array = vectorizer.transform(content);
//			log.info(array.toString());
			jobAd.setArray(array);
			toReturn.add(jobAd);
		}

		return toReturn;

	}

}
