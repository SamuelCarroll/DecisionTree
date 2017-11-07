package main

import (
	"math"
	"readFile"
	"reflect"
)

// Node -- basic node for our tree struct
type Node struct {
	NodeData   []*readFile.Wine
	Leaf       bool
	IndexSplit int
	SplitVal   float64
}

// Tree -- tree structure
type Tree struct {
	Data         Node
	Class        int
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

func initializeAvgs(example readFile.Wine) []interface{} {
	var newAvgVals []interface{}

	for i := range example.FeatureSlice {
		switch example.FeatureSlice[i].(type) {
		case float64:
			newAvgVals = append(newAvgVals, 0.0)
		case bool:
			newAvgVals = append(newAvgVals, false)
		case string:
			newAvgVals = append(newAvgVals, "")
		}
	}

	return newAvgVals
}

func runningAvg(oldAvgs []interface{}, newVal readFile.Wine, n int) []interface{} {
	if len(oldAvgs) < len(newVal.FeatureSlice) {
		oldAvgs = initializeAvgs(newVal)
	}

	for i := range newVal.FeatureSlice {
		//reflect the type of the feature slice index handle float, bool and string (don't worry about bool and str yet)
		switch val := oldAvgs[i].(type) {
		case float64:
			temp := float64(val) * float64(n)
			temp += getFloatReflectVal(newVal.FeatureSlice[i])
			oldAvgs[i] = temp / float64(n+1)
		}
	}

	return oldAvgs
}

func findStds(classSam []*readFile.Wine, class ClassAvg) []interface{} {

	var stdDev []interface{}
	featureLen := len(classSam[0].FeatureSlice)

	for i := 0; i < featureLen; i++ {
		classTotal := 0.0
		for _, sample := range classSam {
			class.stdDev = append(class.stdDev, 0.0)
			//reflect the type of the feature slice index handle float, bool and string (don't worry about bool and str yet)
			fsample := getFloatReflectVal(sample.FeatureSlice[i])
			fclass := getFloatReflectVal(class.averages[i])
			classTotal += math.Pow((fsample - fclass), 2)
		}
		stdDev = append(stdDev, classTotal/float64(class.count))
	}

	return stdDev
}

func findEntropy(valueIndex int, avg, stdDev float64, nodeData []*readFile.Wine) float64 {
	in1, in2, in3 := 0.0, 0.0, 0.0
	e1, w1, e2, w2, e3, w3 := 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

	for _, wine := range nodeData {

		instance := getFloatReflectVal(wine.FeatureSlice[valueIndex])
		if wine.Class == 1 {
			in1 += countClass(instance, avg+stdDev)
		} else if wine.Class == 2 {
			in2 += countClass(instance, avg+stdDev)
		} else if wine.Class == 3 {
			in3 += countClass(instance, avg+stdDev)
		}
	}

	lenData := float64(len(nodeData))

	if in1 > 0 {
		w1 = in1 / lenData
		e1 = (w1) * math.Log2(w1)
	}
	if in2 > 0 {
		w2 = in2 / lenData
		e2 = (w2) * math.Log2(w2)
	}
	if in3 > 0 {
		w3 = in3 / lenData
		e3 = (w3) * math.Log2(w3)
	}

	entropy := w1*e1 + w2*e2 + w3*e3

	return entropy * -1
}

func findLeast(c1, c2, c3 float64) (int, float64) {
	if c1 <= c2 && c1 <= c3 {
		return 0, c1
	} else if c2 <= c1 && c2 <= c3 {
		return 1, c2
	}

	return 2, c3
}

func getMajority(data []*readFile.Wine) int {
	count1, count2, count3 := 0, 0, 0
	for _, wine := range data {
		if wine.Class == 1 {
			count1++
		} else if wine.Class == 2 {
			count2++
		} else {
			count3++
		}
	}
	if count1 > count2 && count1 > count3 {
		return 1
	} else if count2 > count1 && count2 > count3 {
		return 2
	} else {
		return 3
	}
}

func getFloatReflectVal(val interface{}) float64 {
	v := reflect.ValueOf(val)
	v = reflect.Indirect(v)

	floatVal := v.Convert(reflect.TypeOf(0.0))
	return floatVal.Float()
}

func getBoolReflectVal(val interface{}) bool {
	v := reflect.ValueOf(val)
	v = reflect.Indirect(v)

	boolVal := v.Convert(reflect.TypeOf(true))
	return boolVal.Bool()
}

func getStrReflectVal(val interface{}) string {
	v := reflect.ValueOf(val)
	v = reflect.Indirect(v)

	strVal := v.Convert(reflect.TypeOf(""))
	return strVal.String()
}
