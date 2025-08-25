# 📝 Fingerprint Minutiae Matching – Bozorth-like Method (Simple Guide)

## 1. What is the problem?

We want to check if two fingerprint images come from the **same finger**.
Each fingerprint has a set of **minutiae** (small points with position and direction).

The question:
👉 *“Do the minutiae of print A line up with the minutiae of print B?”*



## 2. Why not just compare points directly?

If we look at minutiae one by one, we have problems:

* Fingers can be **rotated** or **shifted** when scanned.
* Some minutiae may be **missing** or **extra** because of noise.

So we need a method that is **robust**: it should still find a match even if the prints are not perfectly aligned.



## 3. The idea of Bozorth3

Instead of matching minutiae directly, Bozorth compares **pairs of minutiae**.

For each pair `(i, j)` in a fingerprint we measure:

* **Distance** between the two points.
* **Relative angles**: how the direction of each minutia relates to the line joining them.

👉 These values are *invariant* – they do not change if we rotate or move the whole fingerprint.



## 4. Step 1 – Build an “intra-table”

For every pair of minutiae in fingerprint A:

* Save `(distance, relative angle of i, relative angle of j)`

Do the same for fingerprint B.
This is like a “catalog” of geometric relationships inside each fingerprint.


## 5. Step 2 – Find compatible pairs

Now, we compare the two catalogs.

If pair `(i, j)` in A and pair `(k, l)` in B have:

* **Similar distance** (within a tolerance)
* **Similar relative angles**

then we say:

* maybe minutia `i` in A corresponds to `k` in B
* and minutia `j` in A corresponds to `l` in B

This gives us possible **correspondences**.



## 6. Step 3 – Build a graph of correspondences

We represent correspondences as a **graph**:

* **Nodes** = “minutia i in A corresponds to minutia k in B”
* **Edges** = two correspondences that support each other (because their pairs were compatible)

So the graph captures **which matches are consistent together**.



## 7. Step 4 – Find the largest consistent cluster

Now we search the graph for a **cluster of nodes** that are:

* Strongly connected (each correspondence agrees with the others)
* One-to-one (a minutia in A can only match one minutia in B)

👉 This cluster represents the set of minutiae that really align between A and B.



## 8. Step 5 – Compute a score

The **size of the cluster** = number of consistent matches.
We turn this into a similarity score, for example:

```
score = 1 – exp( – matches / 8 )
```

* If there are 0 matches → score ≈ 0
* If there are \~10 matches → score ≈ 0.7
* If there are \~20 matches → score ≈ 0.9

So higher score = more likely the same finger.



## 9. Why does this work?

* It ignores simple rotation and translation (because we use relative geometry).
* It tolerates missing or extra minutiae.
* It rewards **global consistency**: not just a few random matches, but a whole *pattern* of minutiae that agree.



## 10. Summary in one sentence

👉 We build a **graph of possible minutiae matches** where edges mean “these matches make sense together”.
The largest consistent group of nodes in this graph gives us the **similarity score** between two fingerprints.


