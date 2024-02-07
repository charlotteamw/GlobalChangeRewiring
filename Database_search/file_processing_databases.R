library(litsearchr)
library(RefManageR)
library(dplyr)
library(readr)
library(stringr)

# Read PubMed results
pubmed_results <- import_results(file="pubmed-coupling-set.nbib")
coupling_pubmed <- pubmed_results %>% 
                    filter(!grepl("Review", publication_type))

# Read WoS results
wos_results <- ReadBib("wos_coupling_1.bib")
wos_results_df1 <- as.data.frame(wos_results)

wos_results2 <- ReadBib("wos_coupling_2.bib")
wos_results_df2 <- as.data.frame(wos_results2)

# Identify the common columns
common_columns <- intersect(names(wos_results_df1), names(wos_results_df2))

# Subset the data frames to only include common columns
wos_results_df1_common <- wos_results_df1[, common_columns]
wos_results_df2_common <- wos_results_df2[, common_columns]

# Merge the two data frames
wos_results_df <- rbind(wos_results_df1_common, wos_results_df2_common)

coupling_wos <- wos_results_df %>% 
                filter(!grepl("Review", type))


# Function to normalize titles
normalize_title <- function(title) {
  words <- str_split(tolower(title), "\\s+") %>%   # Convert to lowercase and split into words
    unlist() %>%                                  # Convert to vector
    str_replace_all("[[:punct:]]", "") %>%        # Remove punctuation
    sort() %>%                                    # Sort words
    paste(collapse = " ")                         # Join back into a single string
  return(words)
}


# Normalize the titles in both datasets
coupling_wos$normalized_title <- sapply(coupling_wos$title, normalize_title)
coupling_pubmed$normalized_title <- sapply(coupling_pubmed$title, normalize_title)

# Find titles in pubmed data not in wos data and filter the rows
rows_not_in_wos <- coupling_pubmed %>%
  filter(!(normalized_title %in% coupling_wos$normalized_title))

# Viewing the result
print(rows_not_in_wos)

# Optionally, save this data to a CSV file
write.csv(rows_not_in_wos, "pubmed_files.csv", row.names = FALSE)


