//Neural Network for NAND Gate, 2 Input Nodes, 1 Hidden Layer w/ 2 Nodes, 1 Output Node

//Sigmoid/Logistic Function on Net Inputs
function sigmoidActivation(netInput){
    return 1 / (1 + (Math.E ** -netInput));
}

//Hidden Layer Total Net Input
function netInput(currentInput1, currentInput2, currentWeight1, currentWeight2, currentBias){
    return (currentInput1 * currentWeight1) + (currentInput2 * currentWeight2) + (currentBias * 1);
}

//Root Mean Square Error
function rootMeanSquareError(target, output){
    return (1/2)*((target - output)**2)
}
//Calculate deltas(partial derivatives) of RMS for each w, = d(RMS)/d(w)
function calculateDelta(deltaNum, output, target, h1Output, h2Output, input1, input2){
    let deltaNodeO1 = (output - target) * output * (1-output);
    let deltaNodeH1 =(deltaNodeO1 * w5) * h1Output * (1 - h1Output);
    let deltaNodeH2 =(deltaNodeO1 * w6) * h2Output * (1 - h2Output);
    if(deltaNum === "w1"){
	return deltaNodeO1 * w5 * h1Output*(1-h1Output) * input1;
    }
    else if(deltaNum === "w2"){
	return deltaNodeO1 * w6 * h2Output*(1-h2Output) * input1;
    }
    else if(deltaNum === "w3"){
	return deltaNodeO1 * w5 * h1Output*(1-h1Output) * input2;
    }
    else if(deltaNum === "w4"){
	return deltaNodeO1 * w6 * h2Output*(1-h2Output) * input2;
    }
    else if(deltaNum === "w5"){
	return deltaNodeO1 * h1Output;
    }
    else if(deltaNum === "w6"){
	return deltaNodeO1 * h2Output;
    }
    else if(deltaNum === "b2"){
	return deltaNodeO1;
    }
    else if(deltaNum === "b1"){
	return deltaNodeH1+deltaNodeH2;
    }
}

//Update Weights from BackPropagation
function updateWeights(weight, learningRate, currentDelta){
    return weight - (learningRate * currentDelta);
}

//Inputs and Target Outputs
let inputs = [[0,0], [0,1], [1,0], [1,1]];
let targetOutputs = [1, 1, 1, 0];

//Initialize Values, b=bias, w=weights, h=hidden node net output,eta=learning rate, guessedOutput=feedForwardOuput,rms=root Mean Square Error
let b1 = 0.5, b2 = 0.5;
let w1 = 0.5, w2 = 0.5, w3 = 0.5, w4 = 0.5, w5 = 0.5, w6 = 0.5;
let h1, h2;
let eta = .8;
let guessedOutput;
let rms;

console.log("Test for sigmoid func., should be ~.75 = "+ sigmoidActivation(1.106));
console.log("Test for RMS func., should be ~0.275 = " + rootMeanSquareError(0.01,0.751));
console.log("Test for NetInput func., should be 0.9 = "+netInput(1,1,.2,.2,.5));
for(let i = 0; i < 100000000; i++){
    for(let j = 0; j < inputs.length; j++){
	let input1 = inputs[j][0];
	let input2 = inputs[j][1];
	let currentTargetOutput = targetOutputs[j];

	//Forward Pass Through Network
	let h1Output = sigmoidActivation(netInput(input1,input2, w1, w3, b1));
	let h2Output = sigmoidActivation(netInput(input1, input2, w2, w4, b1));
	
	guessedOutput = sigmoidActivation(netInput(h1Output, h2Output, w5, w6, b2));
	rms = rootMeanSquareError(currentTargetOutput, guessedOutput);
	if(i%100000 === 0){
	    console.log("Output = "+guessedOutput +"  Target = " + currentTargetOutput + "  Error ="+ rms);
	}

	//Backward Pass Through Network
	
	//Finding the Partial Derivatives of RMS from w's
	let delta1 = calculateDelta("w1",guessedOutput, currentTargetOutput, h1Output, h2Output,input1,input2);
	let delta2 = calculateDelta("w2",guessedOutput, currentTargetOutput, h1Output, h2Output,input1,input2);
	let delta3 = calculateDelta("w3",guessedOutput, currentTargetOutput, h1Output, h2Output,input1,input2);
	let delta4 = calculateDelta("w4",guessedOutput, currentTargetOutput, h1Output, h2Output,input1,input2);
	let delta5 = calculateDelta("w5",guessedOutput, currentTargetOutput, h1Output, h2Output,input1,input2);
	let delta6 = calculateDelta("w6",guessedOutput, currentTargetOutput, h1Output, h2Output,input1,input2);
	let deltaB2 = calculateDelta("b2", guessedOutput, currentTargetOutput, h1Output, h2Output, input1, input2);
	let deltaB1 = calculateDelta("b1", guessedOutput, currentTargetOutput, h1Output, h2Output, input1, input2);

	//Updating New Weights
	w1 = updateWeights(w1, eta, delta1);
	w2 = updateWeights(w2, eta, delta2);
	w3 = updateWeights(w3, eta, delta3);
	w4 = updateWeights(w4, eta, delta4);
	w5 = updateWeights(w5, eta, delta5);
	w6 = updateWeights(w6, eta, delta6);
	b1 = updateWeights(b1, eta, deltaB1);
	b2 = updateWeights(b2, eta, deltaB2);
    }
}