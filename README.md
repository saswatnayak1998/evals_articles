### **Summary: Quantifying Metrics**

1. **Relevance**:

   - Calculated using cosine similarity between query embeddings and response chunk embeddings.
   - Averages the relevance scores of all chunks to provide a total relevance score.

2. **Completeness**:

   - Measures the overlap between tokens in the query and the response.
   - Quantified as the ratio of common tokens to total query tokens.

3. **Overlap**:

   - Assesses redundancy by calculating pairwise cosine similarity between embeddings of response chunks.
   - Higher similarity penalizes the score, with a normalized result: `1 - average similarity`.

4. **Engagement**:

   - How engaging is this text. Can be modified and used for various types of readers(general, specialized etc). Uses GPT4 to evaluate the engagement.

Each metric is designed to be interpretable and actionable, providing insights into the quality, coverage, and uniqueness of the system’s responses. Each one of them is then multiplied with weights and added to get a final_score.

5. What do you think are the most pressing challenges that a company like Capitol would face and how can evaluations help?

   ## **Pressing Challenges for Capitol.ai and Role of Evaluations**

- **Ensuring Output Quality**:

  - **Challenge**: Maintaining relevance, accuracy, and logical structure across multi-modal responses.
  - **Evaluations Help**: Use relevance, coherence, and factual accuracy metrics to ensure high-quality responses.

- **Reducing Redundancy**:

  - **Challenge**: Avoiding repetitive information across chunks.
  - **Evaluations Help**: Use overlap metrics to penalize redundancy and improve uniqueness.

- **Scaling Automated Evaluations**:

  - **Challenge**: Manual evaluation is infeasible for large-scale outputs.
  - **Evaluations Help**: Automate evaluation pipelines for relevance, completeness, and factual validation.

- **Handling Complex Queries**:

  - **Challenge**: Balancing depth, clarity, and personalization in responses.
  - **Evaluations Help**: Use completeness and clarity metrics to address user intent effectively.

2. Among these, pick 1-3 evaluation tasks of your choice that can be largely completed in the time given, and write evaluation scripts for those tasks.
3. Are your results actionable? There's a big difference between an evaluation that makes a directionally correct conclusion and an evaluation that is precise enough for us to use when making client-impacting decisions. Think about what an evaluation needs to do to be used in production.

4. If there is anything you don't have time to do, spend time thinking about what you would do with more time, since I may ask about this.

- To improve this further, I would look at the type of FAQs and see where the system underperforms. I would try to build custom solutions for the majority of such cases.
- I would fine tune the prompt or fine tune the LLM to give desired answers based on the queries. I would try to maximize the retention time of a user as a parameter to be maximized. More the amount of time used in reading the articles, more the revenues.

## **Use something like Activeloop database which is suited for storing and querying multimodal data(since we can have images and videos in the article too). Every data is stored as a tensor and can be retrieved accordingly.**
