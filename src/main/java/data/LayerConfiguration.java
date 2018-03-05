package data;

import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
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
	public static DenseLayer getDenseLayer(int nIn, int nOut, String activ, String weight) {

		DenseLayer layer = new DenseLayer.Builder().nIn(nIn).nOut(nOut).weightInit(getWeight(weight))
				.activation(getActivation(activ)).build();
		return layer;
	}

	public static EmbeddingLayer getEmbeddingLayer(int nIn, int nOut, String activ, String weight) {

		EmbeddingLayer layer = new EmbeddingLayer.Builder().nIn(nIn).nOut(nOut).weightInit(getWeight(weight))
				.activation(getActivation(activ)).build();
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
		if (weight.equals("zero"))
			weightInit = WeightInit.ZERO;

		return weightInit;

	}

}
