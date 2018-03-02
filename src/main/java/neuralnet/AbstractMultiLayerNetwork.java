package neuralnet;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public abstract class AbstractMultiLayerNetwork {
	
	public abstract MultiLayerNetwork buildGraph(int numberOfInputs, int numberOfOutputs);

}
