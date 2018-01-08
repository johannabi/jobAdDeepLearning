package preprocessing;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import org.apache.uima.resource.ResourceInitializationException;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.ListStringSplit;
import org.deeplearning4j.bagofwords.vectorizer.BaseTextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import application.App;
import de.uni_koeln.spinfo.classification.core.data.ClassifyUnit;
import de.uni_koeln.spinfo.ml_classification.data.FocusClassifyUnit;

public class Vectorizer {

	private static Logger log = LoggerFactory.getLogger(App.class);

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

	public void vectorize(List<ClassifyUnit> trainingSet) throws ResourceInitializationException {

		// TokenizerFactory tokFac = new UimaTokenizerFactory();
		//
		// TfidfVectorizer vectorizer = new TfidfVectorizer.Builder()
		// .setMinWordFrequency(1)
		// .setStopWords(new ArrayList<String>())
		// .setTokenizerFactory(tokFac)
		// .setIterator(iter)
		// .allowParallelTokenization(false)
		//// .labels(labels)
		//// .cleanup(true)
		// .build();

//		for (ClassifyUnit cu : trainingSet) {
//
//			DataSet dataSet;
//
//			Boolean inFocus = ((FocusClassifyUnit) cu).getInFocus().get("Webentwicklung");
//			if (inFocus) {
//				dataSet = vectorizer.vectorize(cu.getContent(), "Web");
//			} else
//				dataSet = vectorizer.vectorize(cu.getContent(), "NoWeb");
//
//			System.out.println(dataSet);
//
//			// System.out.println(cu.getContent());
//			// tokFac.
//			Tokenizer tokenizer = tokFac.create(cu.getContent());
//			List<String> tokens = new ArrayList<String>();
//
//			// create tokens list
//			while (tokenizer.hasMoreTokens()) {
//				String token = tokenizer.nextToken();
//				tokens.add(token);
//			}
//
//			System.out.println(".....");
//
//			System.out.println(tokens);
//		}

	}

}
