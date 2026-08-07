# How Merge Sort Works

## Introduction to Merge Sort
Merge sort is a divide-and-conquer algorithm that splits a list of elements into two halves, recursively sorts each half, and then merges them. The purpose of merge sort is to sort a list of elements in a efficient and stable manner, making it suitable for large datasets. The time complexity of merge sort is O(n log n), which is a significant improvement over other sorting algorithms like bubble sort and insertion sort. This makes merge sort a popular choice for many applications, including database sorting and file system organization. By understanding the basics of merge sort, developers can implement this algorithm in their own projects and take advantage of its efficiency. Overall, merge sort is a fundamental algorithm in computer science, and its principles can be applied to a wide range of problems.

## Merge Sort Algorithm
The merge sort algorithm is a popular sorting technique used to arrange elements in a list in a specific order. The process involves two main steps: divide and merge. 
* The divide step involves splitting the list into two halves until each sublist contains only one element. This is done to ensure that each sublist is inherently sorted, as a list with one element is always sorted.
* The merge step involves combining the sublists in a way that maintains their sorted order. This is achieved by comparing elements from each sublist and placing the smaller element first in the merged list.
* The recursion in merge sort is what allows the algorithm to efficiently sort large lists. The divide step is recursive, as it continues to split the list into smaller sublists until the base case is reached (i.e., a list with one element). The merge step is then applied recursively to combine the sorted sublists, resulting in a fully sorted list. By using recursion, merge sort can handle lists of any size, making it a versatile and efficient sorting algorithm.

## Example Use Case
To illustrate the effectiveness of merge sort, let's consider a real-world scenario where we need to sort an array of exam scores. 
* Provide an example array to sort: Suppose we have an array of exam scores: `[64, 34, 25, 12, 22, 11, 90]`.
* Demonstrate the merge sort process: The merge sort algorithm will recursively divide this array into smaller subarrays until each subarray contains only one element, and then merge these subarrays in a sorted manner.
* Discuss the output of the merge sort: After applying the merge sort algorithm to the array `[64, 34, 25, 12, 22, 11, 90]`, the output will be a sorted array: `[11, 12, 22, 25, 34, 64, 90]`. This demonstrates how merge sort can efficiently sort an array of elements in ascending order.
