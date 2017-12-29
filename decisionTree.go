package DecisionTree

import (
	"fmt"
	"io/ioutil"
	"os"
	"strconv"
	"strings"

	"github.com/SamuelCarroll/DataTypes"
)

// Node -- basic node for our tree struct
// it will contain info on if this node is a Leaf
// what is the index of the value we should split on if this is not a Leaf
// The value we should use as our splitting point
// Finally this contains information on which class this will be if it's a leaf
type Node struct {
	Leaf       bool
	IndexSplit int
	SplitVal   float64
	Class      int
}

// Tree -- tree structure
// Details contains information at this particular level of the tree
// Used indicies keeps track of the indexes we've used for splitting
// Left is all nodes that go to the left
// Right is all nodes that go to the right
type Tree struct {
	Details      Node
	usedIndicies []int
	Left         *Tree
	Right        *Tree
}

// ClassAvg -- holds the averages of each class used for finding split
type ClassAvg struct {
	count    int
	averages []interface{}
	stdDev   []interface{}
}

//Train uses the dataset to train a tree for later predicition
func (decTree Tree) Train(trainSet []*dataTypes.Data, setVal, stopCond float64, classesCount int) Tree {
	var setStack [][]*dataTypes.Data
	var treeStack []*Tree

	//Simplify the basic tree structure
	currTree := &decTree
	currSet := trainSet
	treeLen := 1

	//Ensure we have values before continuing, otherwise we get a runtime error
	for treeLen != 0 {
		var classes []ClassAvg
		var classSamples [][]*dataTypes.Data

		//Initialize all the class averages
		for i := 0; i < classesCount; i++ {
			var newClass ClassAvg

			classes = append(classes, newClass)
			classSamples = append(classSamples, *new([]*dataTypes.Data))

			classes[i].count = 0
		}

		//Average all the classes and find the split
		avgClass(currSet, classSamples, classes)
		left, right := currTree.findSplit(currSet, classes, setVal, stopCond, classesCount)

		//Check if we will continue or if we have a leaf node
		if currTree.Details.Leaf == false {
			//Copy the values to the right and the tree to a stack so we don't use
			//recursion add length to tree
			setStack = append(setStack, right)
			treeStack = append(treeStack, currTree.Right)
			currSet = left
			currTree = currTree.Left
			treeLen++
		} else {
			//get the length of the tree and set curr to the last element in the list
			treeLen--

			if treeLen > 0 {
				currTree, treeStack = treeStack[treeLen-1], treeStack[:treeLen-1]
				currSet, setStack = setStack[treeLen-1], setStack[:treeLen-1]
			}
		}
	}

	//Return the entire tree
	return decTree
}

//Test uses the dataset passed in to predict the dataset
func (decTree Tree) Test(allData []*dataTypes.Data) {
	misclassified := 0
	//Print the header so we can see results
	fmt.Printf("+-----------+----------+-------------------------+\n")
	fmt.Printf("| Predicted |  Actual  |           UID           |\n")
	fmt.Printf("+-----------+----------+-------------------------+\n")
	//For each datum in the data range run it through the completed tree
	for _, datum := range allData {
		prediction := decTree.GetClass(*datum)
		//Check if we have misclassified data, increasing misclassified count if we do
		if prediction != datum.Class {
			misclassified++
		}
		//Print that specific datum's classification result
		fmt.Printf("|     %d     |     %d    |", prediction, datum.Class)
		fmt.Printf("   %s   ", datum.UID)

		//This adds a little to the datum's list because it makes it easier to search
		//for anomalous traffic misclassified as normal traffic
		if prediction == 1 && datum.Class == 2 {
			fmt.Printf(" oops")
		}
		fmt.Printf("\n")
	}
	//Print footer and final tree results
	fmt.Printf("+-----------+----------+-------------------------+\n")

	fmt.Printf("%d out of %d wrongly classified\n", misclassified, len(allData))
	fmt.Printf("Misclassified: %f\n", float64(misclassified)/float64(len(allData)))
}

//GetClass returns an int value that refers to the class a value belongs to
func (decTree Tree) GetClass(datum dataTypes.Data) int {
	currNode := decTree.GetTerminalNode(datum)

	return currNode.Details.Class
}

//GetTerminalNode iterates through a tree for a datum and then returns that node
//that datum is classified into
func (decTree Tree) GetTerminalNode(datum dataTypes.Data) *Tree {
	currNode := &decTree

	for currNode.Details.Leaf == false {
		index := currNode.Details.IndexSplit
		testVal := getVal(datum.FeatureSlice[index])
		if testVal <= currNode.Details.SplitVal {
			currNode = currNode.Left
		} else {
			currNode = currNode.Right
		}
	}

	return currNode
}

//WriteTree will save a tree to a file for use later on
func (decTree *Tree) WriteTree(filename string) {
	//Try to open the output file and return an error if one occurs
	file, err := os.Create(filename)
	if err != nil {
		fmt.Println("Error opening output file: ", filename)
		return
	}

	//Start the current node at the root of the tree and initialize the treeStack
	//for iteration
	currNode := decTree
	var treeStack []*Tree

	//Set length of tree equal to 1 (we have a root node) and start iterating through
	//the tree
	treeLen := 1
	for treeLen != 0 {
		file.WriteString(nodeToStr(currNode.Details))

		//As long as we don't have a leaf node we should go left and append the Right
		//node onto our tree stack so we can come back to it later
		if currNode.Details.Leaf == false {
			treeStack = append(treeStack, currNode.Right)
			currNode = currNode.Left
			treeLen++
		} else {
			//reduce the length of the tree and set curr to the last element in the list
			treeLen--

			if treeLen > 0 {
				currNode, treeStack = treeStack[treeLen-1], treeStack[:treeLen-1]
			}
		}
	}

	file.Close()
}

//ReadTree will read a tree from the specified filename
func (decTree *Tree) ReadTree(filename string) error {
	//Try opening the input file and list any errors we may encounter
	file, err := ioutil.ReadFile(filename)
	if err != nil {
		fmt.Println("Error opening input file: ", filename)
		return err
	}

	//read the entire file into memory and split on new lines
	sDat := fmt.Sprintf("%s", file)
	datLines := strings.Split(sDat, "\n")

	//set the current node to the root and initialize values for iterating over
	//the trees
	currNode := decTree
	var treeStack []*Tree
	treeLen := 1
	lastNode := false

	//while we still have lines in a tree file we want to parse the line,
	//adding the data to our current node
	for _, line := range datLines {
		if !lastNode {
			currNode.Details.Leaf, currNode.Details.IndexSplit, currNode.Details.SplitVal, currNode.Details.Class, err = parseLine(line)
			if err != nil {
				return err
			}

			//While we aren't on a leaf node we want to initialize two child nodes
			//Move to the left and continue iterating
			if currNode.Details.Leaf == false {
				currNode.Left = new(Tree)
				currNode.Right = new(Tree)

				treeStack = append(treeStack, currNode.Right)
				currNode = currNode.Left
				treeLen++
			} else {
				//if we are at a leaf node, move to the most recent right child
				treeLen--
				if treeLen > 0 {
					currNode, treeStack = treeStack[treeLen-1], treeStack[:treeLen-1]
				} else {
					lastNode = true
				}
			}
		}
	}

	return nil
}

//parseLine will parse a single line from a tree file
func parseLine(line string) (bool, int, float64, int, error) {
	//Split the line on commas (basically we have a csv)
	lineItem := strings.Split(line, ",")
	if len(lineItem) < 4 {
		return false, 0, 0.0, 0, nil
	}

	//the file structure is bool, int, float, int
	//which corresponds to leaf node, split attribute index, split value, and node class
	leafNode, err := strconv.ParseBool(lineItem[0])
	if err != nil {
		return false, 0, 0.0, 0, err
	}
	splitIndex, err := getRegInt(lineItem[1])
	if err != nil {
		return false, 0, 0.0, 0, err
	}
	splitValue, err := strconv.ParseFloat(lineItem[2], 64)
	if err != nil {
		return false, 0, 0.0, 0, err
	}
	class, err := getRegInt(lineItem[3])
	if err != nil {
		return false, 0, 0.0, 0, err
	}

	return leafNode, splitIndex, splitValue, class, nil
}

//This function will get a base 32 integer from a string value
func getRegInt(line string) (int, error) {
	var retVal int

	i64, err := strconv.ParseInt(line, 10, 32)
	if err != nil {
		return retVal, err
	}

	retVal = int(i64)

	return retVal, nil
}

//nodeToStr will take a node value and return a csv line representation
//of that node for forest storage
func nodeToStr(currNode Node) string {
	leafStr := strconv.FormatBool(currNode.Leaf)
	indexSplit := strconv.Itoa(currNode.IndexSplit)
	splitVal := strconv.FormatFloat(currNode.SplitVal, 'f', 24, 64)
	classStr := strconv.Itoa(currNode.Class)

	return leafStr + "," + indexSplit + "," + splitVal + "," + classStr + "\n"
}

//TODO consider shortening this function!!!
//findSplit will take all the attributes and find the best split value
//however this requires finding and comparing all possible split values
func (decTree *Tree) findSplit(currData []*dataTypes.Data, classes []ClassAvg, setVal, stopCond float64, numClasses int) ([]*dataTypes.Data, []*dataTypes.Data) {
	if stoppingCond(currData, stopCond, numClasses) {
		decTree.Details.Leaf = true
		decTree.Details.Class = getMajority(currData, numClasses)
		return nil, nil
	}

	numFields := len(currData[0].FeatureSlice)

	var splitVals []float64
	var entropys []float64
	var left []*dataTypes.Data
	var right []*dataTypes.Data

	//for each attribute
	//handle the calculation of the entropy for that attribute, needed to find
	//split
	for i := 0; i < numFields; i++ {
		indexUsed := false
		//for each used index initialize the entropy to a huge value and the split to a small value
		for _, temp := range decTree.usedIndicies {
			if temp == i {
				entropys = append(entropys, setVal)
				splitVals = append(splitVals, 0)
				indexUsed = true
			}
		}

		//ensure we haven't used this index before calculating the entrophy
		if indexUsed == false {
			var tempVals []float64
			var averages []float64
			var stdDevs []float64
			var tempEntropys []float64

			//For each class in the classes slice
			for _, class := range classes {
				//if a class is empty we should initialize it
				if len(class.averages) == 0 {
					averages = append(averages, setVal)
					stdDevs = append(stdDevs, setVal)
					tempVals = append(tempVals, setVal)
					tempEntropys = append(tempEntropys, setVal)
				} else {
					//if we have something that is initialized we can append the new values
					//the average attribute value for that class, the standard deviation
					//a proposed split value and the entropy of using that split value
					averages = append(averages, GetFloatReflectVal(class.averages[i]))
					stdDevs = append(stdDevs, GetFloatReflectVal(class.stdDev[i]))
					tempVals = append(tempVals, averages[len(averages)-1]+stdDevs[len(stdDevs)-1])
					tempEntropys = append(tempEntropys, findEntropy(i, len(classes), averages[len(averages)-1], stdDevs[len(stdDevs)-1], currData))
				}
			}

			//Find which class has the better split value
			tempIndex, tempEntropy := findLeast(tempEntropys)

			//add that entropy and split value to our list for later use
			splitVals = append(splitVals, tempVals[tempIndex])
			entropys = append(entropys, tempEntropy)
		}
	}

	//Here we want to find the smallest entropy to use in the
	index := findIndex(entropys)

	//Initialize the node values
	decTree.Details.Leaf = false
	decTree.Details.SplitVal = splitVals[index]
	decTree.Details.IndexSplit = index

	//create new children nodes
	decTree.Left = new(Tree)
	decTree.Right = new(Tree)

	//Add the index to the used indicies list
	decTree.Left.usedIndicies = append(decTree.usedIndicies, decTree.Details.IndexSplit)
	decTree.Right.usedIndicies = append(decTree.usedIndicies, decTree.Details.IndexSplit)

	for _, elem := range currData {
		compVal := getVal(elem.FeatureSlice[index])

		if compVal <= splitVals[index] {
			left = append(left, elem)
		} else {
			right = append(right, elem)
		}
	}

	//Decided if we have a good split, if all values go left or right we should end
	if len(left) == len(currData) {
		decTree.Details.Leaf = true
		decTree.Details.Class = getMajority(currData, numClasses)
		left, right = nil, nil
	} else if len(right) == len(currData) {
		decTree.Details.Leaf = true
		decTree.Details.Class = getMajority(currData, numClasses)
		left, right = nil, nil
	}

	return left, right
}

//getVal will get a value from an abstract interface type
func getVal(val interface{}) float64 {
	testVal := 0.0
	//switch on the detected type of variable, currently we support
	//float64 and bool values
	switch val.(type) {
	case float64:
		testVal = GetFloatReflectVal(val)
	case bool:
		testVal = GetBoolReflectVal(val)
	}

	return testVal
}
