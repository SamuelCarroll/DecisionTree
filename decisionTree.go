package DecisionTree

import (
	"fmt"

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

var class1 ClassAvg
var class2 ClassAvg
var class3 ClassAvg

var class1Sample []*dataTypes.Data
var class2Sample []*dataTypes.Data
var class3Sample []*dataTypes.Data

//Train uses the dataset to train a tree for later predicition
func (decTree Tree) Train(trainSet []*dataTypes.Data, setVal, stopCond float64) Tree {
	class1.count = 0
	class2.count = 0
	class3.count = 0
	var setStack [][]*dataTypes.Data
	var treeStack []*Tree

	currTree := &decTree
	currSet := trainSet
	treeLen := 1

	for treeLen != 0 {
		avgClass(currSet)
		left, right := currTree.findSplit(currSet, setVal, stopCond)

		if currTree.Details.Leaf == false {
			setStack = append(setStack, right)
			treeStack = append(treeStack, currTree.Right)
			currSet = left
			currTree = currTree.Left
			treeLen++
		} else {
			//get the length of the tree and set curr to the last element in the list
			treeLen--

			if treeLen-1 >= 0 {
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

func (decTree *Tree) findSplit(currData []*dataTypes.Data, setVal, stopCond float64) ([]*dataTypes.Data, []*dataTypes.Data) {
	if stoppingCond(currData, stopCond) {
		decTree.Details.Leaf = true
		decTree.Details.Class = getMajority(currData)
		return nil, nil
	}

	numFields := len(currData[0].FeatureSlice)

	var splitVals []float64
	var entropys []float64

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
			// we want to get the average of the interface intance
			class1Avg := getFloatReflectVal(class1.averages[i])
			class1Std := getFloatReflectVal(class1.stdDev[i])

			class2Avg := getFloatReflectVal(class2.averages[i])
			class2Std := getFloatReflectVal(class2.stdDev[i])

			class3Avg := getFloatReflectVal(class3.averages[i])
			class3Std := getFloatReflectVal(class3.stdDev[i])

			c1 := findEntropy(i, class1Avg, class1Std, currData)
			c2 := findEntropy(i, class2Avg, class2Std, currData)
			c3 := findEntropy(i, class3Avg, class3Std, currData)

			//find best split for that attribute
			tempVals = append(tempVals, class1Avg+class1Std, class2Avg+class2Std, class3Avg+class3Std)

			tempIndex, tempEntropy := findLeast(c1, c2, c3)

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

	var left []*dataTypes.Data
	var right []*dataTypes.Data

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
