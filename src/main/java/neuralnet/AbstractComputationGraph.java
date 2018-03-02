package neuralnet;

import org.deeplearning4j.nn.graph.ComputationGraph;

public abstract class AbstractComputationGraph {
	
	
	public abstract ComputationGraph buildGraph(int numberOfInputs, int numberOfOutputs);

}
