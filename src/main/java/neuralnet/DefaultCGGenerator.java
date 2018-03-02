package neuralnet;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jfree.util.Log;
import org.nd4j.linalg.activations.Activation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import workflow.Workflow;

public class DefaultCGGenerator extends AbstractComputationGraph{

	private static Logger log = LoggerFactory.getLogger(DefaultCGGenerator.class);
	private int firstHiddenNodes = 150;
	private int secondHiddenNodes = 50;
	private double learningRate = 0.001;
	
	public DefaultCGGenerator(int firstHiddenNodes, int secondHiddenNodes,
			double learningRate) {
		this.firstHiddenNodes = firstHiddenNodes;
		this.secondHiddenNodes = secondHiddenNodes;
		this.learningRate = learningRate;
	}
	
	public DefaultCGGenerator() {
		
	}
	
	@Override
	public ComputationGraph buildGraph(int numberOfInputs, int numberOfOutputs) {
		
		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(learningRate) 
                .graphBuilder()
                .addInputs("input")            
//                .addLayer("Conv", new ConvolutionLayer.Builder()
//                		.nIn(firstHiddenNodes)
//                		.nOut(secondHiddenNodes)
//                		.activation(Activation.RELU)
//                		.build(), "input")
                .addLayer("Dense1", new DenseLayer.Builder()
                		.nIn(numberOfInputs)
                		.nOut(firstHiddenNodes)
                		.activation(Activation.SIGMOID) //(Activation.RELU)
                		.build(), "input")
                .addLayer("Dense2", new DenseLayer.Builder()
                		.nIn(firstHiddenNodes)
                		.nOut(secondHiddenNodes)
                		.activation(Activation.SIGMOID) //(Activation.RELU)
                		.build(), "Dense1") 
//                .addLayer("RNN", new RnnOutputLayer.Builder()
//                		.nIn(secondHiddenNodes)
//                		.nOut(numberOfOutputs)
//                		.activation(Activation.SIGMOID)
//                		.build(), "Dense2")
                .addLayer("Output", new OutputLayer.Builder()
                		.nIn(secondHiddenNodes)
                		.nOut(numberOfOutputs)
                		.build(), "Dense2")
                .setOutputs("Output")
                .backprop(true)
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        return net;
	}

}
