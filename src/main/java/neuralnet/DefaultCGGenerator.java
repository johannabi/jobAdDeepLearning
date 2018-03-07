package neuralnet;

import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.jfree.util.Log;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import workflow.Workflow;

public class DefaultCGGenerator extends AbstractComputationGraph {

	private static Logger log = LoggerFactory.getLogger(DefaultCGGenerator.class);
	// private int firstHiddenNodes = 150;
	// private int secondHiddenNodes = 50;
	private double learningRate = 0.001;
	private boolean backprop = true;
	private Map<String, FeedForwardLayer> layers = new HashMap<String, FeedForwardLayer>();
	private int seed = 123;

	/**
	 * sets the configuration for building a neural net
	 * @param learningRate
	 * @param backprop
	 * @param layers
	 */
	public DefaultCGGenerator(double learningRate, boolean backprop,
			Map<String, FeedForwardLayer> layers) {
		this.learningRate = learningRate;
		this.backprop = backprop;
		this.layers = layers;
	}

	public DefaultCGGenerator(Map<String, FeedForwardLayer> layers) {
		this.layers = layers;
	}
	

	/**
	 * builds the graph with the configurations (of the constructor) with the given
	 * number of input and output nodes
	 */
	@Override
	public ComputationGraph buildGraph(int numberOfInputs, int numberOfOutputs) {

		GraphBuilder builder = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.learningRate(learningRate)
				.graphBuilder();

		String currentInput = "input";
//		int currentNOut = 0;

		builder.addInputs(currentInput);
		for (Map.Entry<String, FeedForwardLayer> e : layers.entrySet()) {
			if (e.getValue().getNIn() == -1)
				e.getValue().setNIn(numberOfInputs);
			if (e.getValue().getNOut() == -2)
				e.getValue().setNOut(numberOfOutputs);
			builder.addLayer(e.getKey(), e.getValue(), currentInput);
			currentInput = e.getKey();
//			currentNOut = e.getValue().getNOut();
		}

		ComputationGraphConfiguration conf = builder		
				.setOutputs("Output")
				.backprop(backprop)
				.build();

		ComputationGraph net = new ComputationGraph(conf);
		return net;
	}

}
