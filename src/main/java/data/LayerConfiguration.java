package data;

import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
import org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;

public class LayerConfiguration {

	/**
	 * creates a Dense Layer with given parameters
	 * @param nIn number of input nodes
	 * @param nOut number of output nodes
	 * @param activ type of activation (relu, sigmoid, softmax)
	 * @param weight type of initial weight (xavier, zero)
	 * @return
	 */
	public static DenseLayer getDenseLayer(int nIn, int nOut, Activation activ, WeightInit weight) {

		DenseLayer layer = new DenseLayer.Builder().nIn(nIn).nOut(nOut).weightInit(weight)
				.activation(activ).build();
		return layer;
	}
	
	public static CenterLossOutputLayer getCenterLossOutputLayer(int nIn, int nOut, WeightInit weight) {		
		CenterLossOutputLayer layer = new CenterLossOutputLayer.Builder()
				.nIn(nIn)
				.nOut(nOut)	
				.weightInit(weight)
				.build();
		
		return layer;
	}
	
	public static OutputLayer getOutputLayer(int nIn, int nOut) {
		OutputLayer layer = new OutputLayer.Builder()
				.nIn(nIn)
				.nOut(nOut)
				.build();
		
		return layer;
	}

	public static EmbeddingLayer getEmbeddingLayer(int nIn, int nOut, Activation activation, WeightInit weight) {

		EmbeddingLayer layer = new EmbeddingLayer.Builder()
				.nIn(nIn)
				.nOut(nOut)
				.weightInit(weight)
				.activation(activation)
				.build();
		return layer;
	}

	private static Activation getActivation(String activ) {
		Activation activation = null;

		if (activ.equals("relu"))
			activation = Activation.RELU;
		else if (activ.equals("sigmoid"))
			activation = Activation.SIGMOID;
		else if (activ.equals("softmax"))
			activation = Activation.SOFTMAX;

		return activation;
	}

	private static WeightInit getWeight(String weight) {
		WeightInit weightInit = null;

		if (weight.equals("xavier"))
			weightInit = WeightInit.XAVIER;
		else if (weight.equals("zero"))
			weightInit = WeightInit.ZERO;
		else if (weight.equals("distribution"))
			weightInit = WeightInit.DISTRIBUTION;
		else if (weight.equals("uniform"))
			weightInit = WeightInit.UNIFORM;
		else if (weight.equals("relu"))
			weightInit = WeightInit.RELU;

		return weightInit;

	}

	

}
