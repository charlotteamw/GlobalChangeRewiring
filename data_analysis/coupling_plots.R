library(ggplot2)
library(tidyverse)
library(patchwork)
library(knitr)
library(kableExtra)

df_coupling <- read.csv("coupling_synthesis.csv")

df_coupling$result <- as.factor(df_coupling$result)
df_coupling$ecosystem <- as.factor(df_coupling$ecosystem)
df_coupling$stressor <- as.factor(df_coupling$stressor)


## Aquatic/Terrestrial Table
aqu_terr_percentages <- aqu_terr_counts %>%
  mutate(percentage = n / sum(n) * 100)

names(aqu_terr_percentages) <- c("Aquatic/Terrestrial", "n", "% of Studies")

kable(aqu_terr_percentages, caption = "Percentage of Aquatic and Terrestrial Studies") %>%
  kable_styling(bootstrap_options = c("striped", "hover"))


# donut 

donut_df <- df_coupling %>%
  count(result) %>%
  mutate(fraction = (n / sum(n)))

donut_df$ymax = cumsum(donut_df$fraction)
donut_df$ymin = c(0, head(donut_df$ymax, n=-1))

donut_plot <- ggplot(donut_df, aes(ymax=ymax, ymin=ymin, xmax=4, xmin=3, fill=result)) +
  geom_rect()+
  coord_polar(theta="y")+ 
  xlim(c(2, 4))+
  theme_void() +
  geom_text(aes(x = 3.5, label = paste0(round(fraction * 100, 1), "%"), y = (ymin + ymax) / 2), color = "white", size = 5)+
  scale_fill_manual(values = c("steelblue4", "skyblue3", "grey71")) + 
  labs(fill = "Result")+
  theme(legend.title = element_text(size = 14),
        legend.text = element_text(size = 12)) 

donut_plot

# mechanism

mechanism_df <-df_coupling %>%
  count(mechanism, result) %>%
  filter(!is.na(mechanism) & !is.na(result))%>%
  mutate(result = factor(result, levels = c("Increase", "Decrease")))

mechanism_plot <- ggplot(mechanism_df, aes(x = mechanism, y = n, fill = result)) + 
  geom_bar(position = "stack", stat = "identity") +
  scale_fill_manual(values = c("Decrease" = "steelblue4", "Increase" = "skyblue3")) +
  ylim(0, 40) +
  theme_classic()+ 
  labs(fill = "Result", y = "Count", x = "Mechanism")+
  theme(legend.title = element_text(size = 16),
        legend.text = element_text(size = 14), 
        axis.text = element_text(size = 14), 
        axis.title = element_text(size = 16)) 

mechanism_plot

# donut & mechanism plot 

donut_plot <- donut_plot + theme(plot.margin = margin(r = 5, unit = "mm")) # Add right margin to donut plot
mechanism_plot <- mechanism_plot + theme(plot.margin = margin(l = 5, unit = "mm"), legend.position = "none") # Add left margin to mechanism plot and remove its legend

donut_mechanism_combined <- donut_plot + mechanism_plot + 
  plot_layout(widths = c(1.0, 0.8)) # Increase the relative size of the donut plot and decrease the mechanism plot

donut_mechanism_combined

#ecosystem

ecosystem_df <- df_coupling %>%
  count(ecosystem, result) %>%
  mutate(result = factor(result, levels = c("No change", "Increase", "Decrease")))

ecosystem_totals <- ecosystem_df %>%
  group_by(ecosystem) %>%
  summarise(total_n = sum(n))

ecosystem_df <- ecosystem_df %>%
  left_join(ecosystem_totals, by = "ecosystem")

ecosystem_plot <- ggplot(ecosystem_df, aes(x = reorder(ecosystem, -total_n), y = n, fill = result)) +
  geom_bar(stat = "identity", position = "stack") +
  scale_fill_manual(values = c("Decrease" = "steelblue4", "Increase" = "skyblue3", "No change" = "grey71")) +
  ylim(0, 20) +
  theme_classic() +
  labs(fill = "Result", y = "Count", x = "Ecosystem") +
  theme(legend.title = element_text(size = 16),
        legend.text = element_text(size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 14), # Rotate x-axis labels to prevent overlap
        axis.text.y = element_text(size = 14),
        axis.title = element_text(size = 16))

ecosystem_plot

# stressor

stressor_df <- df_coupling %>%
  count(stressor, result) %>%
  mutate(result = factor(result, levels = c("No change", "Increase", "Decrease")))

stressor_totals <- stressor_df %>%
  group_by(stressor) %>%
  summarise(total_n = sum(n))

stressor_df <- stressor_df %>%
  left_join(stressor_totals, by = "stressor")

stressor_plot <- ggplot(stressor_df, aes(x = reorder(stressor, -total_n), y = n, fill = result)) +
  geom_bar(stat = "identity", position = "stack") +
  scale_fill_manual(values = c("Decrease" = "steelblue4", "Increase" = "skyblue3", "No change" = "grey71")) +
  ylim(0, 15) +
  theme_classic() +
  labs(fill = "Result", y = "Count", x = "Stressor") +
  theme(legend.title = element_text(size = 16),
        legend.text = element_text(size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 14), # Rotate x-axis labels to prevent overlap
        axis.text.y = element_text(size = 14),
        axis.title = element_text(size = 16))

stressor_plot

# ecosystem & stressor plot

ecosystem_plot <- ecosystem_plot + theme(plot.margin = margin(r = 10, unit = "mm")) # Add right margin to ecosystem plot
stressor_plot <- stressor_plot + theme(plot.margin = margin(l = 10, unit = "mm")) # Add left margin to stressor plot

combined_plot <- ecosystem_plot + stressor_plot + plot_layout(guides = "collect") 

combined_plot



# Calculate the counts for each combination of ecosystem, stressor, and result
count_df <- df_coupling %>%
  group_by(ecosystem, stressor, result) %>%
  summarize(count = n(), .groups = 'drop')

# Calculate the order for ecosystem and stressor by the number of studies
ecosystem_order <- count_df %>%
  group_by(ecosystem) %>%
  summarize(total_count = sum(count)) %>%
  arrange(desc(total_count)) %>%
  mutate(ecosystem = factor(ecosystem, levels = rev(ecosystem)))

stressor_order <- count_df %>%
  group_by(stressor) %>%
  summarize(total_count = sum(count)) %>%
  arrange(desc(total_count)) %>%
  mutate(stressor = factor(stressor, levels = rev(stressor)))

# Join the order information back to the original count_df
count_df <- count_df %>%
  left_join(ecosystem_order, by = "ecosystem") %>%
  left_join(stressor_order, by = "stressor") %>%
  arrange(desc(count))  # Arrange in descending order to plot larger bubbles first

# Create the bubble plot with the ordered factors
bubble_plot <- ggplot(count_df, aes(x = ecosystem, y = stressor, size = count, color = result)) +
  geom_point() +  # removed alpha for non-translucent bubbles
  scale_size(range = c(4, 15)) +  # Adjust the size range as needed
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y = element_text(angle = 45, vjust = 1)) +
  labs(title = "Bubble Plot of Ecosystem, Stressor, and Results",
       x = "Ecosystem Type",
       y = "Anthropogenic Pressure",
       size = "Count",
       color = "Result") +
  scale_color_manual(values = c("Decrease" = "steelblue4", "Increase" = "skyblue3", "No change" = "grey71")) +
  scale_x_discrete(limits = ecosystem_order$ecosystem) +
  scale_y_discrete(limits = stressor_order$stressor) +
  guides(size = guide_legend(reverse = TRUE))  # Reverse the legend for bubble sizes

# Print the bubble plot
print(bubble_plot)

