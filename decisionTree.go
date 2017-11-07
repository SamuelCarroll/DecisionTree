package main

import (
	"fmt"
	"readFile"
)

var class1 ClassAvg
var class2 ClassAvg
var class3 ClassAvg

var class1Sample []*readFile.Wine
var class2Sample []*readFile.Wine
var class3Sample []*readFile.Wine

var setVal float64
var stopCond float64

func main() {
	var decTree Tree
	setVal = 100000000000.0
	stopCond = 0.84
	decTree.Data.NodeData = readFile.Read("/home/ritadev/Documents/Thesis_Work/Decision-Tree/trainwine.data")
	fmt.Printf("Error Rate: %f\n", stopCond)

	class1.count = 0
	class2.count = 0
	class3.count = 0

	decTree = decTree.train()
	testWines := readFile.Read("/home/ritadev/Documents/Thesis_Work/Decision-Tree/winetest.data")

	decTree.test(testWines)
}

func (decTree Tree) train() Tree {
	var treeStack []*Tree

	curr := &decTree
	treeLen := 1

	for treeLen != 0 {
		avgClass(curr.Data.NodeData)
		curr.findSplit()

		if curr.Data.Leaf == false {
			treeStack = append(treeStack, curr.Right)
			curr = curr.Left
			treeLen++
		} else {
			//get the length of the tree and set curr to the last element in the list
			treeLen--

			if treeLen-1 >= 0 {
				curr, treeStack = treeStack[treeLen-1], treeStack[:treeLen-1]
			}
		}
	}

	fmt.Println("tree built")
	return decTree
}

func (decTree Tree) test(Wines []*readFile.Wine) {
	misclassified := 0
	fmt.Printf("+-----------+----------+\n")
	fmt.Printf("| Predicted |  Actual  |\n")
	fmt.Printf("+-----------+----------+\n")
	for _, wine := range Wines {
		prediction := getClass(decTree, *wine)
		fmt.Printf("|     %d     |     %d    |\n", prediction, wine.Class)
		if prediction != wine.Class {
			misclassified++
		}
	}
	fmt.Printf("+-----------+----------+\n")

	fmt.Printf("%d out of %d wrongly classified\n", misclassified, len(Wines))
	fmt.Printf("Misclassified: %f\n", float64(misclassified)/float64(len(Wines)))
}

func getClass(decTree Tree, wine readFile.Wine) int {
	currNode := &decTree

	for currNode.Data.Leaf == false {
		index := currNode.Data.IndexSplit
		testVal := getFloatReflectVal(wine.FeatureSlice[index])
		if testVal < currNode.Data.SplitVal {
			currNode = currNode.Left
		} else {
			currNode = currNode.Right
		}
	}

	return currNode.Class
}

func countClass(instance float64, splitVal float64) float64 {
	if instance < splitVal {
		return 1
	}

	return 0
}

func stoppingCond(nodeData []*readFile.Wine) bool {
	var count [3]int
	var percent [3]float64

	for _, elem := range nodeData {
		count[elem.Class-1]++
	}

	for i := range count {
		percent[i] = float64(count[i]) / float64(len(nodeData))
		if percent[i] >= stopCond {
			return true
		}
	}

	return false
}

func (decTree *Tree) findSplit() {
	if stoppingCond(decTree.Data.NodeData) {
		decTree.Data.Leaf = true
		decTree.Class = getMajority(decTree.Data.NodeData)
		return
	}

	numFields := len(decTree.Data.NodeData[0].FeatureSlice)

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

			c1 := findEntropy(i, class1Avg, class1Std, decTree.Data.NodeData)
			c2 := findEntropy(i, class2Avg, class2Std, decTree.Data.NodeData)
			c3 := findEntropy(i, class3Avg, class3Std, decTree.Data.NodeData)

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
	decTree.Data.Leaf = false
	decTree.Data.SplitVal = splitVals[index]
	decTree.Data.IndexSplit = index

	decTree.Left = new(Tree)
	decTree.Right = new(Tree)

	decTree.Left.usedIndicies = append(decTree.usedIndicies, decTree.Data.IndexSplit)
	decTree.Right.usedIndicies = append(decTree.usedIndicies, decTree.Data.IndexSplit)

	for _, elem := range decTree.Data.NodeData {
		compVal := getFloatReflectVal(elem.FeatureSlice[index])

		if compVal < splitVals[index] {
			decTree.Left.Data.NodeData = append(decTree.Left.Data.NodeData, elem)
		} else {
			decTree.Right.Data.NodeData = append(decTree.Right.Data.NodeData, elem)
		}
	}

	if len(decTree.Left.Data.NodeData) == len(decTree.Data.NodeData) {
		decTree.Data.Leaf = true
		decTree.Class = getMajority(decTree.Data.NodeData)
	} else if len(decTree.Right.Data.NodeData) == len(decTree.Data.NodeData) {
		decTree.Data.Leaf = true
		decTree.Class = getMajority(decTree.Data.NodeData)
	}
}

func findIndex(entropyVals []float64) int {
	minVal := entropyVals[0]
	minIndex := 0

	for i, contender := range entropyVals {
		if contender < minVal {
			minIndex = i
			minVal = contender
		}
	}

	return minIndex
}

func avgClass(wines []*readFile.Wine) {
	for _, wine := range wines {
		if wine.Class == 1 {
			class1.averages = runningAvg(class1.averages, *wine, class1.count)
			class1.count++
			class1Sample = append(class1Sample, wine)
		} else if wine.Class == 2 {
			class2.averages = runningAvg(class2.averages, *wine, class2.count)
			class2.count++
			class2Sample = append(class2Sample, wine)
		} else if wine.Class == 3 {
			class3.averages = runningAvg(class3.averages, *wine, class3.count)
			class3.count++
			class3Sample = append(class3Sample, wine)
		}
	}
	class1.stdDev = findStds(class1Sample, class1)
	class2.stdDev = findStds(class2Sample, class2)
	class3.stdDev = findStds(class3Sample, class3)
}
