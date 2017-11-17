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
type Node struct {
	Leaf       bool
	IndexSplit int
	SplitVal   float64
	Class      int
}

// Tree -- tree structure
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

	currTree := &decTree
	currSet := trainSet
	treeLen := 1

	for treeLen != 0 {
		var classes []ClassAvg
		var classSamples [][]*dataTypes.Data

		for i := 0; i < classesCount; i++ {
			var newClass ClassAvg

			classes = append(classes, newClass)
			classSamples = append(classSamples, *new([]*dataTypes.Data))

			classes[i].count = 0
		}

		avgClass(currSet, classSamples, classes)
		left, right := currTree.findSplit(currSet, classes, setVal, stopCond)

		if currTree.Details.Leaf == false {
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

	return decTree
}

//Test uses the dataset passed in to predict the dataset
func (decTree Tree) Test(allData []*dataTypes.Data) {
	misclassified := 0
	fmt.Printf("+-----------+----------+\n")
	fmt.Printf("| Predicted |  Actual  |\n")
	fmt.Printf("+-----------+----------+\n")
	for _, datum := range allData {
		prediction := decTree.GetClass(*datum)
		if prediction != datum.Class {
			misclassified++
		}
		fmt.Printf("|     %d     |     %d    |\n", prediction, datum.Class)
	}
	fmt.Printf("+-----------+----------+\n")

	fmt.Printf("%d out of %d wrongly classified\n", misclassified, len(allData))
	fmt.Printf("Misclassified: %f\n", float64(misclassified)/float64(len(allData)))
}

//GetClass returns an int value that refers to the class a value belongs to
func (decTree Tree) GetClass(datum dataTypes.Data) int {
	currNode := &decTree

	for currNode.Details.Leaf == false {
		index := currNode.Details.IndexSplit
		testVal := getFloatReflectVal(datum.FeatureSlice[index])
		if testVal < currNode.Details.SplitVal {
			currNode = currNode.Left
		} else {
			currNode = currNode.Right
		}
	}

	return currNode.Details.Class
}

//WriteTree will save a tree to a file for use later on
func (decTree *Tree) WriteTree(filename string) {
	file, err := os.Create(filename)
	if err != nil {
		fmt.Println("Error opening output file: ", filename)
		return
	}

	currNode := decTree
	var treeStack []*Tree

	treeLen := 1
	for treeLen != 0 {
		file.WriteString(nodeToStr(currNode.Details))

		if currNode.Details.Leaf == false {
			treeStack = append(treeStack, currNode.Right)
			currNode = currNode.Left
			treeLen++
		} else {
			//get the length of the tree and set curr to the last element in the list
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
	file, err := ioutil.ReadFile(filename)
	if err != nil {
		fmt.Println("Error opening input file: ", filename)
		return err
	}

	sDat := fmt.Sprintf("%s", file)
	datLines := strings.Split(sDat, "\n")

	currNode := decTree
	var treeStack []*Tree
	treeLen := 1
	lastNode := false

	for _, line := range datLines {
		if !lastNode {
			currNode.Details.Leaf, currNode.Details.IndexSplit, currNode.Details.SplitVal, currNode.Details.Class, err = parseLine(line)
			if err != nil {
				return err
			}

			if currNode.Details.Leaf == false {
				currNode.Left = new(Tree)
				currNode.Right = new(Tree)

				treeStack = append(treeStack, currNode.Right)
				currNode = currNode.Left
				treeLen++
			} else {
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

func parseLine(line string) (bool, int, float64, int, error) {
	lineItem := strings.Split(line, ",")
	if len(lineItem) < 4 {
		return false, 0, 0.0, 0, nil
	}

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

func getRegInt(line string) (int, error) {
	var retVal int

	i64, err := strconv.ParseInt(line, 10, 32)
	if err != nil {
		return retVal, err
	}

	retVal = int(i64)

	return retVal, nil
}

func nodeToStr(currNode Node) string {
	leafStr := strconv.FormatBool(currNode.Leaf)
	indexSplit := strconv.Itoa(currNode.IndexSplit)
	splitVal := strconv.FormatFloat(currNode.SplitVal, 'f', 24, 64)
	classStr := strconv.Itoa(currNode.Class)

	return leafStr + "," + indexSplit + "," + splitVal + "," + classStr + "\n"
}

//TODO shorten this function!!!
func (decTree *Tree) findSplit(currData []*dataTypes.Data, classes []ClassAvg, setVal, stopCond float64) ([]*dataTypes.Data, []*dataTypes.Data) {
	if stoppingCond(currData, stopCond) {
		decTree.Details.Leaf = true
		decTree.Details.Class = getMajority(currData)
		return nil, nil
	}

	numFields := len(currData[0].FeatureSlice)

	var splitVals []float64
	var entropys []float64
	var left []*dataTypes.Data
	var right []*dataTypes.Data

	//for each attribute
	for i := 0; i < numFields; i++ {
		indexUsed := false
		for _, temp := range decTree.usedIndicies {
			if temp == i {
				entropys = append(entropys, setVal)
				splitVals = append(splitVals, 0)
				indexUsed = true
			}
		}

		if indexUsed == false {
			var tempVals []float64
			var averages []float64
			var stdDevs []float64
			var tempEntropys []float64

			for _, class := range classes {
				if len(class.averages) == 0 {
					averages = append(averages, setVal)
					stdDevs = append(stdDevs, setVal)
					tempVals = append(tempVals, setVal)
					tempEntropys = append(tempEntropys, setVal)
				} else {
					averages = append(averages, getFloatReflectVal(class.averages[i]))
					stdDevs = append(stdDevs, getFloatReflectVal(class.stdDev[i]))
					tempVals = append(tempVals, averages[len(averages)-1]+stdDevs[len(stdDevs)-1])
					tempEntropys = append(tempEntropys, findEntropy(i, len(classes), averages[len(averages)-1], stdDevs[len(stdDevs)-1], currData))
				}
			}

			// TODO modify to take unspecified number of classes using a slice
			tempIndex, tempEntropy := findLeast(tempEntropys)

			//Here we have a problem, we are appending the entropy not the value to split on
			splitVals = append(splitVals, tempVals[tempIndex])
			entropys = append(entropys, tempEntropy)
		}
	}

	index := findIndex(entropys)

	//don't use entropy as your stopping condition, find a way to measure the purity after a split
	decTree.Details.Leaf = false
	decTree.Details.SplitVal = splitVals[index]
	decTree.Details.IndexSplit = index

	decTree.Left = new(Tree)
	decTree.Right = new(Tree)

	decTree.Left.usedIndicies = append(decTree.usedIndicies, decTree.Details.IndexSplit)
	decTree.Right.usedIndicies = append(decTree.usedIndicies, decTree.Details.IndexSplit)

	for _, elem := range currData {
		compVal := getFloatReflectVal(elem.FeatureSlice[index])

		if compVal < splitVals[index] {
			left = append(left, elem)
		} else {
			right = append(right, elem)
		}
	}

	if len(left) == len(currData) {
		decTree.Details.Leaf = true
		decTree.Details.Class = getMajority(currData)
		left, right = nil, nil
	} else if len(right) == len(currData) {
		decTree.Details.Leaf = true
		decTree.Details.Class = getMajority(currData)
		left, right = nil, nil
	}

	return left, right
}
