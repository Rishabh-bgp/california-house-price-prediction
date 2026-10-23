# FAQ
**Why can it not predict above $500k?** The data top-codes every value at or above $500k, so nothing above it can be learned.

**Why are values so low?** They are 1990 dollars for whole block groups.

**Why is spatial CV worse?** Random folds let neighbouring areas leak; spatial CV holds whole regions out.
