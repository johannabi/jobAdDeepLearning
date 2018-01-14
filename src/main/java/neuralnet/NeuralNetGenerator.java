package neuralnet;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;

public class NeuralNetGenerator {
	
	/**
	 * generates default neural net just to test if workflow is practical.
	 * hidden layer contains 1500 nodes
	 * @param numberOfInputs number of input nodes
	 * @param numberOfOutputs number of output nodes (one for every label)
	 * @return neural net (computation graph) with hidden layer (dense layer) and ouput layer
	 */
	public static ComputationGraph generateDefaultNet(int numberOfInputs, int numberOfOutputs) {
		int numberOfHiddenNodes = 1500;
		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.01)
                .graphBuilder()
                .addInputs("input")  
                .addLayer("L1", new DenseLayer.Builder()
                		.nIn(numberOfInputs)
                		.nOut(numberOfHiddenNodes)
                		.build(), "input")            
                .addLayer("L2", new OutputLayer.Builder()
                		.nIn(numberOfHiddenNodes)
                		.nOut(numberOfOutputs)
                		.build(), "L1")
                .setOutputs("L2")	
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        return net;
	}

}
