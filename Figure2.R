# Input folder numbering differs from the manuscript/output numbering.
script_dir <- local({
 files <- vapply(sys.frames(), function(f) if(is.null(f$ofile)) "" else f$ofile, character(1))
 files <- files[nzchar(files)]
 arg <- grep("^--file=",commandArgs(),value=TRUE)
 if(length(files)) dirname(normalizePath(tail(files,1))) else if(length(arg)) dirname(normalizePath(sub("^--file=","",arg[1]))) else getwd()
})
source(file.path(script_dir,"figure_plot_utils.R"))
suppressPackageStartupMessages(library(patchwork))
figure_panels <- list()
# Updated inputs: cool Viridis for 4a2; custom palette from 201-figure4a.R for 4a3.
region_levels <- c("Top-left", "Bottom-left", "Bottom-middle")
region_labels <- c("Top-left"="A", "Bottom-left"="B", "Bottom-middle"="C")
d <- read_panel("Figure2", "figure4a2_IL2_STAT5_SIGNALING") |>
 mutate(region=factor(region, levels=region_levels))
stopifnot(!anyNA(d$region))
# Increase this value to make Figure 4a2's points larger (e.g. 1.2).
figure4a2_point_size <- 1.5
if (!requireNamespace("ggrastr", quietly=TRUE)) {
 stop("Figure 4a2 requires ggrastr. Install it with install.packages('ggrastr').")
}
d <- spatial_coordinates_um(d, "x", "y", "region")
p <- ggplot(d, aes(x_um, y_um, colour=score)) +
 ggrastr::rasterise(geom_point(size=figure4a2_point_size, stroke=0), dpi=300) +
 facet_wrap(~region, ncol=1, labeller=as_labeller(region_labels)) +
 coord_equal() + scale_colour_viridis_c(option="viridis", name="ssGSEA") +
 labs(x=NULL, y=NULL) + style() +
 theme(legend.position="right", axis.text.x=element_blank(),
       axis.text.y=element_blank(), axis.ticks=element_blank())
p <- add_spatial_scale_bar(p, d, "region", length_um=200, label="200 µm")
figure_panels[["figure4a2"]] <- save_panel(p,"figure4a2",d)

d <- read_panel("Figure2", "figure4a3_top_pathway") |>
 mutate(region=factor(region, levels=region_levels))
stopifnot(!anyNA(d$region))
pathway_colours <- setNames(
 colorRampPalette(c("#D1495B", "#EDAE49", "#4C8DB8", "#72B7A1", "#8E6C88"))(
  length(unique(d$pathway))),
 sort(unique(d$pathway)))
p <- ggplot(d, aes(mean_score,
                  tidytext::reorder_within(pathway, mean_score, region), fill=pathway)) +
 geom_col() +
 geom_text(aes(label=sprintf("%.2f", mean_score),
               hjust=if_else(mean_score < 0, 1.15, -.15)),
           colour="black", size=3, show.legend=FALSE) +
 facet_wrap(~region, scales="free_y", ncol=1, labeller=as_labeller(region_labels)) +
 tidytext::scale_y_reordered() +
 scale_fill_manual(values=pathway_colours, name="Pathway",
  labels=function(x) str_wrap(str_replace_all(str_remove(x, "^ssgsea_HALLMARK_"), "_", " "), 28)) +
 scale_x_continuous(expand=expansion(mult=c(.15,.18))) +
 labs(x="Mean ssGSEA score", y=NULL) + style() +
 theme(legend.position="right", axis.text.y=element_blank(), axis.ticks.y=element_blank())
figure_panels[["figure4a3"]] <- save_panel(p,"figure4a3",d)
d <- read_panel("Figure2","2b") |>
 mutate(Run=ordered(Run),
        gene_type=factor(gene_type, levels=c("SVG", "Non-SVG")),
        Panel=factor(paste(gene_set, gene_type),
                     levels=c("Top 20 SVG", "Top 50 SVG", "Top 20 Non-SVG", "Top 50 Non-SVG")))
stopifnot(!anyNA(d$Panel), all(d$q25<=d$median_r), all(d$median_r<=d$q75))
# Each gene type gets its own figure and y-scale; Top 20/50 share that scale.
figure4b_data <- d
for (gene_type_name in c("SVG", "Non-SVG")) {
 d <- figure4b_data |> filter(gene_type == gene_type_name)
 panel_name <- if (gene_type_name == "SVG") "figure4b_svg" else "figure4b_nonsvg"
 p <- ggplot(d, aes(Run, median_r, colour=gene_type, group=gene_type)) +
  geom_line() + geom_point() +
  geom_errorbar(aes(ymin=q25, ymax=q75), width=.15) +
  facet_wrap(~Panel, nrow=1, scales="fixed") +
  scale_colour_manual(values=c(SVG="#D1495B", `Non-SVG`="#4C8DB8")) +
  labs(x=NULL, y="Median Pearson r\non validation cells", colour=NULL) +
  style() + theme(legend.position="none")
 figure_panels[[panel_name]] <- save_panel(p, panel_name, d, tag="b")
}
# Keep the overview file and complete Figure 4, preserving the two separate scales.
figure_panels[["figure4b"]] <-
 (figure_panels[["figure4b_svg"]] |
  (figure_panels[["figure4b_nonsvg"]] + labs(tag=NULL))) + plot_layout(widths=c(1,1))
ggsave(file.path(output_dir, "figure4b.pdf"), figure_panels[["figure4b"]],
       width=panel_sizes[["figure4b"]][1], height=panel_sizes[["figure4b"]][2],
       units="in", device=cairo_pdf, bg="white")
message("Saved figure4b.pdf with separate SVG and non-SVG y-scales")
d <- read_panel("Figure2","2c")
d$method <- factor(d$method,levels=c("ZeroShot",setdiff(unique(d$method),"ZeroShot")))
p <- ggplot(d,aes(method,pearson_r,colour=label,group=label))+geom_line()+geom_point()+labs(x=NULL,y="Pathway Pearson correlation",colour=NULL)+style()
# Original ER pathway colors from 21-figure4c.R.
p <- p + scale_colour_manual(
 values=c("ER early"="#EDA04B", "ER late"="#72B7A1"), name=NULL)
figure_panels[["figure4c"]] <- save_panel(p,"figure4c",d)

# ---- Combine this figure with patchwork ----
# Edit the rows and relative sizes below to adjust the combined layout.
# The spatial map and pathway bars form panel a: show its letter only once.
figure4_row1 <- (figure_panels[["figure4a2"]] |
                (figure_panels[["figure4a3"]] + labs(tag=NULL))) +
  plot_layout(widths=c(4,8))
figure4_row2 <- wrap_plots(figure_panels[c("figure4b", "figure4c")],
                          nrow=1, widths=c(16,5))
figure4_combined <- (figure4_row1 / figure4_row2) +
  plot_layout(heights=c(10,4), guides="keep")

ggsave(file.path(output_dir, "figure4_combined.pdf"), figure4_combined,
       width=13.55, height=14.55, units="in", device=cairo_pdf,
       bg="white", limitsize=FALSE)
message("Saved figure4_combined.pdf with patchwork")
