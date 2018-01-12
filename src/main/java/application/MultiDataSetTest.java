package application;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import evaluation.Evaluator;

public class MultiDataSetTest {
	
	private static Logger log = LoggerFactory.getLogger(CSVTest.class);

	public static void main(String[] args) throws IOException, InterruptedException {
		
		
//		analyseCSV(new File("src/main/resources/data/featurevectors.txt"));
		// 8328 Einträge pro Zeile (12 Labels), 3655 Zeilen (inkl. head)
		
		//First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 1;
        char delimiter = ',';
        RecordReader rrTraining = new CSVRecordReader(numLinesToSkip,delimiter);
        rrTraining.initialize(new FileSplit(new File("src/main/resources/data/crossvalidation/trainingvectors2.txt")));

        RecordReader rrTest = new CSVRecordReader(numLinesToSkip, delimiter);
        rrTest.initialize(new FileSplit(new File("src/main/resources/data/crossvalidation/testvectors1.txt")));
        
        
        int batchSizeTrain = 3285;
        int batchSizeTest = 365;

        int inputFrom = 0;
        int inputTo = 8315;
        int outputFrom = 8316;
        int outputTo = 8327;
        
        
        
        MultiDataSetIterator trainingIter = new RecordReaderMultiDataSetIterator.Builder(batchSizeTrain)
        		.addReader("Reader", rrTraining)
        		.addInput("Reader", inputFrom, inputTo)
        		.addOutput("Reader", outputFrom, outputTo)
        		.build();
        
        MultiDataSetIterator testIter = new RecordReaderMultiDataSetIterator.Builder(batchSizeTest)
        		.addReader("Reader", rrTest)
        		.addInput("Reader", inputFrom, inputTo)
        		.addOutput("Reader", outputFrom, outputTo)
        		.build();
        
        MultiDataSet trainingData = trainingIter.next();
        MultiDataSet testData = testIter.next();
//        log.info(allData.asList().toString());
//        log.info(allData.getLabels().length + " Labels");
//        log.info(allData.getLabels()[0].toString());
//        log.info(allData.getFeatures()[0].toString());
        // TODO https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/seq2seq/AdditionRNN.java
        
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.01)
                .graphBuilder()
                .addInputs("input") //can use any label for this
//                .addLayer("L1", new GravesLSTM.Builder().nIn(inputTo+1).nOut(1500).build(), "input")
                .addLayer("L1", new DenseLayer.Builder().nIn(inputTo+1).nOut(1500).build(), "input")            
                .addLayer("L2", new OutputLayer.Builder().nIn(1500).nOut(12).build(), "L1")
//                .addLayer("L3",new RnnOutputLayer.Builder().nIn(1500).nOut(12).build(), /*"input",*/ "L1")
                .setOutputs("L2")	//We need to specify the network outputs and their order
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        net.fit(trainingData);
        
        //klassifikation
        INDArray out = net.output(testData.getFeatures())[0];
        INDArray features = testData.getFeatures()[0];
        INDArray gold = testData.getLabels()[0];
        
        Evaluator eval = new Evaluator();
        eval.evaluate(features, out, gold);
        
        for(int i = 0;i < batchSizeTest; i++) {
        	log.info("Output" + i + ": " + out.getRow(i).toString());
        	 log.info("Gold" + i + ": " + gold.getRow(i).toString());
        }
        
        //TODO evaluation
        
        
       
//        log.info(out.toString());
        
//        eval.eval(testData.getLabels()[0], out[0]);
//        log.info(eval.stats());
//        
//        
//        Evaluation evaluation = net.evaluate(testIter);
//        log.info(evaluation.stats());

        
        
	}

	private static void analyseCSV(File file) throws IOException {

		BufferedReader br = new BufferedReader(new FileReader(file));
		String line = "";
		int lines = 0;
		while ((line = br.readLine()) != null) {
			List<Double> features = new ArrayList<Double>();
			List<Integer> labeles = new ArrayList<Integer>();
			String entries[] = line.split(",");
			log.info(entries.length + " Einträge pro Zeile");
			lines++;
		}
		log.info(lines + " Zeilen");
	}

}
