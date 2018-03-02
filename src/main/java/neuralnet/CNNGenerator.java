package neuralnet;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class CNNGenerator extends AbstractMultiLayerNetwork {
	
	protected static int channels = 3;	

	
	@Override
	public MultiLayerNetwork buildGraph(int numberOfInputs, int numberOfOutputs) {
		int seed = 6;
		int iterations = 10;
		int height = 0;
		int width = 0;
		channels = numberOfInputs;
		
			
			MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				    .seed(seed)
				    .iterations(iterations)
				    .regularization(false).l2(0.005) 
				    .activation(Activation.RELU)
				    .learningRate(0.0001)
				    .weightInit(WeightInit.XAVIER)
				    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				    .updater(new Nesterovs(0.9))
				    .list()
				    .layer(0, convInit("cnn1", channels, 50 ,  new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
				    .layer(1, maxPool("maxpool1", new int[]{2,2}))
				    .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
				    .layer(3, maxPool("maxool2", new int[]{2,2}))
				    .layer(4, new DenseLayer.Builder().nOut(500).build())
				    .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
				        .nOut(numberOfOutputs)
				        .activation(Activation.SOFTMAX)
				        .build())
				    .backprop(true).pretrain(false)
				    .setInputType(InputType.convolutional(height, width, channels))
				    .build(); 
			
			MultiLayerNetwork net = new MultiLayerNetwork(conf);
			
			
			return net;
	}
	
	private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
	    return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
	}

	private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
	    return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
	}

	private SubsamplingLayer maxPool(String name,  int[] kernel) {
	    return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
	}




}
