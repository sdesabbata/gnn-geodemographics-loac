library(tidyverse)
library(GGally)
library(ggrepel)
library(patchwork)
library(xtable)

rm(list = ls())

evaluation_summary <- 
  read_csv(
    "Analysis/compare-geodemo-evaluation-summary--log.csv",
    col_names = c("Method", "measure", "value")
  ) %>% 
  pivot_wider(
    id_cols = Method,
    names_from = measure,
    values_from = value
  ) %>% 
  add_row(
    `Method` = "LOAC",
    `Matching OAs` = 25053, 
    `SED 167 z-scores` = 114.5,
    `SED 60 k-vars` = 0.834,
    `Spatially clustered OAs` = 15251
  ) %>% 
  left_join(
    read_csv("Analysis/compare-geodemo-evaluation-summary--info.csv")
  ) %>% 
  mutate(
    Design = 
      factor(Design,
        levels = c("KM", "SFCM", "Vec", "vGNN", "NAGAE-d1", "NAGAE-d2")
     ),
    `Spatial graph` =
      factor(`Spatial graph`,
             levels = c("None", "Queens", "MDT", "KNN8")
     ),
    Aggregator =
      factor(Aggregator,
             levels = c("None", "Mean", "MeanPooling", "MaxPooling", "Att")
      ),
    Input =
      factor(Input,
             levels = c("60 k-vars", "167 z-scores", "60 PCA")
      )
  ) %>% 
  mutate(
    Approach = 
      factor(paste0(Design, " (", Input, ")"),
             levels = c("KM (60 k-vars)", "KM (60 PCA)", "SFCM (60 k-vars)", "SFCM (60 PCA)", "Vec (167 z-scores)", "vGNN (60 k-vars)", "vGNN (167 z-scores)", "NAGAE-d1 (167 z-scores)", "NAGAE-d2 (167 z-scores)")
      )
  ) %>% 
  mutate(
    `Matching OAs (pct)` = (`Matching OAs` / 25053) * 100,
    `Spatially clustered OAs (pct)` = (`Spatially clustered OAs` / 25053) * 100
  ) %>% 
  mutate(
    plot_label = case_when(
      Method == "LOAC" ~ "Fig. 3a",
      Method == "Queens_SFCM_60pca" ~ "Fig. 3b",
      Method == "KNN8_GCN_167zscores" ~ "Fig. 3c",
      Method == "Queens_NAGAE_d1" ~ "Fig. 3d",
      Method == "KNN8_GSonly_MeanAgg_167zscores" ~ "Fig. 3e",
      Method == "Queens_NAGAE_d2" ~ "Fig. 3f",
    )
  )

evaluation_summary %>% 
  write_csv("Analysis/compare-geodemo-evaluation-summary--full.csv")

annotation_arrow <- 
  arrow(
    length = unit(0.1, "inches"),
    type = "closed"
  )

summary_plot_matching <- 
  evaluation_summary %>% 
  #filter(Method != "LOAC") %>% 
  ggplot(aes(
    x = `Spatially clustered OAs (pct)`,
    y = `Matching OAs (pct)`,
    fill = Approach, 
    shape = `Spatial graph`,
    label = plot_label
    #color = Input,
    # stroke =  case_when(
    #   Input == "60 k-vars" ~ 0.8, 
    #   Input == "167 z-scores" ~ 1.2, 
    #   Input == "60 PCA" ~ 1.6
    # )
  )) +
  geom_point(size = 3) +
  geom_text_repel(size = 3, force = 1, force_pull = 0.1, box.padding = 2, min.segment.length = 0, segment.curvature = -0.1, seed = 16) + 
  #scale_fill_brewer(palette = "Set1", guide = guide_legend(override.aes = list(shape = 21))) + 
  #scale_fill_manual(values = c("#e31a1c", "#fb9a99", "#ff7f00", "#fdbf6f", "#cab2d6", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c"), guide = guide_legend(override.aes = list(shape = 21))) + 
  scale_fill_manual(values = c("#117733", "#44AA99", "#999933", "#DDCC77", "#AA4499", "#CC6677", "#882255", "#332288", "#88CCEE"), guide = guide_legend(override.aes = list(shape = 21))) + 
  #scale_color_manual(values = c("#000000", "#666666","#CCCCCC")) +
  scale_shape_manual(values=c(21, 24, 22, 23)) +
  guides(
    color = guide_legend(
      override.aes=list(shape = 0, stroke = c(0.8, 1.2, 1.6))
    )
  ) +
  # Limit y axis to remove LOAC from plot
  # without disrupting the legend
  ylim(30, 80) +
  ylab("OAs matching LOAC") +
  theme_bw()

summary_plot_SED167z <- 
  evaluation_summary %>% 
  ggplot(aes(
    x = `Spatially clustered OAs (pct)`,
    y = `SED 167 z-scores`,
    fill = Approach, 
    shape = `Spatial graph`,
    label = plot_label
    #color = Input,
    # stroke =  case_when(
    #   Input == "60 k-vars" ~ 0.8, 
    #   Input == "167 z-scores" ~ 1.2, 
    #   Input == "60 PCA" ~ 1.6
    # )
  )) +
  geom_point(size = 3) +
  geom_text_repel(size = 3, force = 1, force_pull = 0.1, box.padding = 2, min.segment.length = 0, segment.curvature = -0.1, seed = 16) + 
  #scale_fill_brewer(palette = "Set1", guide = guide_legend(override.aes = list(shape = 21))) + 
  #scale_fill_manual(values = c("#e31a1c", "#fb9a99", "#ff7f00", "#fdbf6f", "#cab2d6", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c"), guide = guide_legend(override.aes = list(shape = 21))) + 
  scale_fill_manual(values = c("#117733", "#44AA99", "#999933", "#DDCC77", "#AA4499", "#CC6677", "#882255", "#332288", "#88CCEE"), guide = guide_legend(override.aes = list(shape = 21))) + 
  #scale_color_manual(values = c("#000000", "#666666", "#CCCCCC")) +
  scale_shape_manual(values=c(21, 24, 22, 23)) +
  guides(
    color = guide_legend(
      override.aes=list(shape = 0, stroke = c(0.8, 1.2, 1.6))
    )
  ) +
  annotate(
    "text", 
    x = 96.5, y = 113.5, 
    label = "Better results",
    size = 3,
    angle = -38
  ) +
  annotate(
    "segment", 
    x = 92, xend = 98, 
    y = 114, yend = 108,
    colour = "black",
    arrow = annotation_arrow
  ) +
  theme_bw()

summary_plot_SED60k <- 
  evaluation_summary %>% 
  ggplot(aes(
    x = `Spatially clustered OAs (pct)`,
    y = `SED 60 k-vars`,
    fill = Approach, 
    shape = `Spatial graph`,
    label = plot_label
    #color = Input,
    # stroke =  case_when(
    #   Input == "60 k-vars" ~ 0.8, 
    #   Input == "167 z-scores" ~ 1.2, 
    #   Input == "60 PCA" ~ 1.6
    # )
  )) +
  geom_point(size = 3) +
  geom_text_repel(size = 3, force = 1, force_pull = 0.1, box.padding = 2, min.segment.length = 0, segment.curvature = -0.1, seed = 16) + 
  #scale_fill_brewer(palette = "Set1", guide = guide_legend(override.aes = list(shape = 21))) + 
  #scale_fill_manual(values = c("#e31a1c", "#fb9a99", "#ff7f00", "#fdbf6f", "#cab2d6", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c"), guide = guide_legend(override.aes = list(shape = 21))) + 
  scale_fill_manual(values = c("#117733", "#44AA99", "#999933", "#DDCC77", "#AA4499", "#CC6677", "#882255", "#332288", "#88CCEE"), guide = guide_legend(override.aes = list(shape = 21))) + 
  #scale_color_manual(values = c("#000000", "#666666", "#CCCCCC")) +
  scale_shape_manual(values=c(21, 24, 22, 23)) +
  guides(
    color = guide_legend(
      override.aes=list(shape = 0, stroke = c(0.8, 1.2, 1.6))
    )
  ) +
  annotate(
    "text", 
    x = 96.5, y = 0.91, 
    label = "Better results",
    size = 3,
    angle = -40
  ) +
  annotate(
    "segment", 
    x = 92, xend = 98, 
    y = 0.92, yend = 0.86,
    colour = "black",
    arrow = annotation_arrow
  ) +
  theme_bw()

summary_plots <-
  (summary_plot_SED60k / summary_plot_SED167z / summary_plot_matching) +
  plot_annotation(
    tag_levels = c("a"),
  ) + 
  plot_layout(
    guides = 'collect'
  )

ggsave(
  filename = "Images/compare-geodemos_matching_SED_spatial-clust.png",
  plot = summary_plots,
  width = 200,
  height = 240,
  units = "mm"
  )


ggsave(
  filename = "Images/compare-geodemos_matching_SED_spatial-clust.pdf",
  plot = summary_plots,
  width = 200,
  height = 240,
  units = "mm"
)

evaluation_summary %>% 
  dplyr::select(
    Approach, Design, Input, `Spatial graph`, Aggregator,
    `SED 167 z-scores`, `SED 60 k-vars`, `Spatially clustered OAs (pct)`, `Matching OAs (pct)`
  ) %>% 
  xtable(type = "latex") %>% 
  print(file = "Backup/Analysis/compare-geodemos_matching_SED_spatial-clust.tex")