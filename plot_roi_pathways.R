suppressPackageStartupMessages({
  library(ggplot2)
  library(SingleCellExperiment)
  library(scater)
  library(SpatialExperiment)
  library(Matrix)
})

enrichment=readRDS("../out/sceGT_enrichment_ssgsea.rds")
sce.pred=readRDS("../out/breastcancer_slides2_sce.rds")



head(colnames(enrichment))
head(colData(sce.pred)$cell_id)
head(colnames(sce.pred))

# how many match each option?
sum(colnames(enrichment) %in% colData(sce.pred)$cell_id)
sum(colnames(enrichment) %in% colnames(sce.pred))


cell_ids <- as.character(colData(sce.pred)$cell_id)
enr_ids  <- as.character(colnames(enrichment))

idx <- match(cell_ids, enr_ids)   # where each spe cell appears in enrichment

keep <- !is.na(idx)
spe2 <- sce.pred[, keep]
enr2 <- enrichment[, idx[keep], drop = FALSE]

stopifnot(identical(as.character(colData(spe2)$cell_id), as.character(colnames(enr2))))


enr_df <- as.data.frame(t(as.matrix(enr2)))  # cells × pathways
colnames(enr_df) <- paste0("ssgsea_", make.names(colnames(enr_df)))

# bind into colData
colData(spe2) <- S4Vectors::DataFrame(colData(spe2), enr_df)




pathway <- "ssgsea_HALLMARK_IL2_STAT5_SIGNALING"  # change this

df <- data.frame(
  x = spatialCoords(spe2)[, "x"],
  y = spatialCoords(spe2)[, "y"],
  score = colData(spe2)[[pathway]],
  sample_id = spe2$sample_id
)

ggplot(df, aes(x = x, y = y, color = score)) +
  geom_point(size = 0.3) +
  # coord_equal() +
  scale_y_reverse() +                 # often needed for image-like orientation
  facet_wrap(~ sample_id) +           # remove if you only want one sample
  labs(title = pathway, color = "SSGSEA") +
  theme_bw()






coords <- spatialCoords(spe2)
coords <- as.matrix(coords)                 # just to be safe
rownames(coords) <- colnames(spe2)          # REQUIRED

reducedDim(spe2, "spatial") <- coords


plotReducedDim(
  spe2,
  dimred = "spatial",
  colour_by = pathway,
  point_size=0.5,
) +
  scale_y_reverse()





library(ggplot2)
library(tidyr)
library(dplyr)
library(viridis)

# select all ssgsea pathways
pathways <- grep("^ssgsea_", colnames(colData(spe2)), value = TRUE)

df <- data.frame(
  x = spatialCoords(spe2)[, "x"],
  y = spatialCoords(spe2)[, "y"],
  colData(spe2)[, pathways, drop = FALSE]
) |>
  pivot_longer(
    cols = all_of(pathways),
    names_to = "pathway",
    values_to = "score"
  )
sel=c("ssgsea_HALLMARK_ADIPOGENESIS","ssgsea_HALLMARK_ALLOGRAFT_REJECTION")
pa=ggplot(df[df$pathway %in% sel, ],
       aes(x = x, y = y, color = score)) +
  geom_point(size = 0.3) +
  scale_y_reverse() +
  scale_color_viridis_c(
    option = "viridis",
    name = "ssGSEA",
    limits = quantile(df$score[df$pathway %in% sel],
                      probs = c(0.05, 0.95),
                      na.rm = TRUE),
    oob = scales::squish
  ) +
  facet_wrap(~ pathway, scales = "free") +
  theme_bw() +
  theme(
    strip.text = element_text(size = 8),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  )


pb=ggplot(df[df$pathway %in% sel, ],
          aes(x = x, y = y, color = score)) +
  geom_point(size = 0.3) +
  scale_y_reverse() +
  scale_color_viridis_c(
    option = "viridis",
    name = "ssGSEA",
    limits = quantile(df$score[df$pathway %in% sel],
                      probs = c(0.05, 0.95),
                      na.rm = TRUE),
    oob = scales::squish
  ) +
  facet_wrap(~ pathway, scales = "free") +
  theme_bw() +
  theme(
    strip.text = element_text(size = 8),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  )



pc=ggplot(df[df$pathway %in% sel, ],
          aes(x = x, y = y, color = score)) +
  geom_point(size = 0.3) +
  scale_y_reverse() +
  scale_color_viridis_c(
    option = "viridis",
    name = "ssGSEA",
    limits = quantile(df$score[df$pathway %in% sel],
                      probs = c(0.05, 0.95),
                      na.rm = TRUE),
    oob = scales::squish
  ) +
  facet_wrap(~ pathway, scales = "free") +
  theme_bw() +
  theme(
    strip.text = element_text(size = 8),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  )





library(ggplot2)
library(dplyr)

pathway <- "ssgsea_HALLMARK_IL2_STAT5_SIGNALING"

df <- data.frame(
  x = spatialCoords(spe2)[, "x"],
  y = spatialCoords(spe2)[, "y"],
  score = colData(spe2)[[pathway]]
)

# Define your 3 boxes
boxes <- tibble::tribble(
  ~region,          ~x_min, ~y_min, ~x_max, ~y_max,
  "Top-left",         9450, 8650, 12150, 11350,
  "Bottom-left",      7800, 15350, 10400, 18000,
  "Bottom-middle",    14950, 16100, 17600, 18700
)

boxes <- tibble::tribble(
  ~region,          ~x_min, ~x_max, ~y_min, ~y_max,
  "Top-left",         8100,  11500,   8200,  11000,
  "Bottom-left",      6300,   9700,  15500,  18200,
  "Bottom-middle",   14100,  17500,  16300,  19000
)

# Assign each point to a region if it falls inside a box
df_roi <- df %>%
  mutate(row_id = row_number()) %>%
  tidyr::crossing(boxes) %>%
  filter(x >= x_min, x <= x_max, y >= y_min, y <= y_max) %>%
  # keep only needed columns (and avoid duplicates if boxes overlap)
  distinct(row_id, region, .keep_all = TRUE)

figure4a2=ggplot(df_roi, aes(x = x, y = y, color = score)) +
  geom_point(size = 0.8) +
  scale_y_reverse() +
  # coord_equal() +
  facet_wrap(~region,scale="free",ncol=1) +
  scale_color_viridis_c(option = "viridis", name = "ssGSEA") +
  #
  theme_bw() +
  theme(
    strip.text = element_text(size = 10),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  ) +
  labs(title = pathway, x = NULL, y = NULL)



write.csv(df_roi,"../out/figure4a2_IL2_STAT5_SIGNALING.csv",row.names = F,quote = F)

ggsave(figure4a2,filename="../out/figure4a2.pdf",width = 4.5,height = 10)


library(dplyr)
library(tidyr)
library(ggplot2)

# 1) make long df of all pathways
pathways <- grep("^ssgsea_", colnames(colData(spe2)), value = TRUE)

df_long <- data.frame(
  x = spatialCoords(spe2)[, "x"],
  y = spatialCoords(spe2)[, "y"],
  colData(spe2)[, pathways, drop = FALSE]
) |>
  mutate(cell_id = seq_len(n())) |>
  pivot_longer(cols = all_of(pathways),
               names_to = "pathway",
               values_to = "score")

# 2) map cells -> regions using your boxes (keeps only cells inside any box)
cells_in_region <- data.frame(
  cell_id = seq_len(ncol(spe2)),
  x = spatialCoords(spe2)[, "x"],
  y = spatialCoords(spe2)[, "y"]
) |>
  tidyr::crossing(boxes) |>
  filter(x >= x_min, x <= x_max, y >= y_min, y <= y_max) |>
  distinct(cell_id, region)

# 3) mean score per region & pathway
region_pathway_mean <- df_long |>
  inner_join(cells_in_region, by = "cell_id") |>
  group_by(region, pathway) |>
  summarise(
    mean_score = median(score, na.rm = TRUE),
    n_cells = dplyr::n(),
    .groups = "drop"
  )


library(dplyr)
library(tidyr)
library(ggplot2)
library(tidytext)

# top N per region (change N here)
N <- 10

topN <- region_pathway_mean %>%
  group_by(region) %>%
  slice_max(mean_score, n = N, with_ties = FALSE) %>%
  ungroup()
#
# ggplot(topN, aes(x = mean_score,
#                  y = reorder_within(pathway, mean_score, region))) +
#   geom_col() +
#   facet_wrap(~ region, scales = "free_y") +
#   scale_y_reordered() +
#   theme_bw() +
#   labs(x = "Mean ssGSEA score", y = NULL,
#        title = paste0("Top ", N, " pathways per region"))

figure4a3=ggplot(
  topN,
  aes(
    x = mean_score,
    y = reorder_within(pathway, mean_score, region),
    fill = pathway
  )
) +
  geom_col() +
  # geom_text(
  #   aes(label = sprintf("%.3f",mean_score)),
  #   hjust = -0.1,          # push text to the right of the bar
  #   size = 3
  # ) +
  facet_wrap(~ region, scales = "free_y",ncol=1) +
  scale_y_reordered() +
  # scale_fill_viridis_d(option = "viridis") +
  theme_bw() +
  theme(
    axis.text.y  = element_blank(),
    axis.ticks.y = element_blank()
  ) +
  labs(
    x = "Median ssGSEA score",
    y = NULL,
    fill = "Pathway",
    title = paste0("Top ", N, " pathways per region")
  ) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15)))

write.csv(topN,"../out/figure4a3_top_pathway.csv",row.names = F,quote = F)


ggsave(figure4a3,filename="../out/figure4a3.pdf",width = 8,height = 10)
