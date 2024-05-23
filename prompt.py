def get_prompt(questions):
    return f'''
        You are an assistant tasked with summarizing images for retrieval. \
        These summaries will be embedded and used to retrieve the raw image. \
        Give a concise summary of the image that is well optimized for retrieval.
        If it's a table, extract all elements of the table.
        If it's a graph, explain the findings in the graph.
        Do not include any numbers that are not mentioned in the image.

        If there is multiple image and asked of similarity checks. \
        Look at the content of the images, colors, size.

        After analysing summerise the answer with percentage.

        Question:
        {questions}?

        Answer:
    '''