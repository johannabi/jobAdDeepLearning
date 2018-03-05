package data;

import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;

/**
 * stores the configurations of a neural net
 * @author Johanna
 *
 */
public class NeuralNetConfiguration {
	
//	private int firstHiddenNodes;
//	private int secondHiddenNodes;
	private double learningRate;
	private boolean backprop;
	private Map<String, FeedForwardLayer> layers = new HashMap<String, FeedForwardLayer>();
	
	public NeuralNetConfiguration(double learningRate, boolean backprop, 
			Map<String, FeedForwardLayer> layers) {
		
		this.learningRate = learningRate;
		this.backprop = backprop;
		this.layers = layers;
	}

	public boolean getBackprop() {
		return backprop;
	}

	public double getLearningRate() {
		return learningRate;
	}
	
	public Map<String, FeedForwardLayer> getLayers() {
		return layers;
	}

	@Override
	public String toString() {
		StringBuffer buff = new StringBuffer();
		buff.append("_learningrate=" + learningRate);
		buff.append("_backpr=" + backprop);
		for(Map.Entry<String, FeedForwardLayer> e : layers.entrySet()) {
			FeedForwardLayer l = e.getValue();
			buff.append("_layer=" + l.getClass().getSimpleName()
									+ "_nIn=" + l.getNIn()
									+ "_nOut=" + l.getNOut()
									+ "_act=" + l.getActivationFn());
		}
		return buff.toString();
	}

}
