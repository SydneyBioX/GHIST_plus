suppressPackageStartupMessages(library(tidyverse))
# Entry scripts set script_dir, supporting both source() and Rscript.
root <- script_dir
output_dir <- file.path(root, "out", "revision")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
read_panel <- function(folder, panel) {
 d <- read_csv(file.path(root,"in",folder,paste0(panel,".csv")),show_col_types=FALSE)
 if(nrow(problems(d))) stop("CSV parsing problems: ",panel)
 d
}
ordered <- function(x) factor(x,levels=unique(x))
method_colours <- c("GHIST+"="#D1495B","GHIST"="#EDA04B","SpatialEx"="#4C8DB8",Phoenix="#C7D9B7",PASTA="#72B7A1","SpatialEx+"="#B07AA1","Full model"="#D1495B","VQ composition off"="#72B7A1","Randomized VQ composition"="#4C8DB8","Single slide"="#D1495B","Pan-cancer"="#4C8DB8")
# Shared typography and named model palette for every revised figure.
# Full model is the GHIST+ alias in the VQ ablation panel.
style <- function() theme_bw(base_size = 14) + theme(
 text = element_text(colour = "black"),
 panel.grid = element_blank(),
 strip.background = element_rect(fill = "white"),
 strip.text = element_text(size = 18, face = "bold", colour = "black"),
 axis.text = element_text(size = 16, colour = "black"),
 axis.text.x = element_text(size = 16, colour = "black", angle = 60, hjust = 1),
 axis.text.y = element_text(size = 16, colour = "black"),
 axis.title = element_text(size = 18, colour = "black"),
 axis.title.x = element_text(size = 18, colour = "black"),
 axis.title.y = element_text(size = 18, colour = "black"),
 legend.text = element_text(size = 16, colour = "black"),
 legend.title = element_text(size = 18, colour = "black"),
 legend.position = "top")
colours <- function() scale_fill_manual(
 values = method_colours, aesthetics = c("fill", "colour"), name = NULL,
 labels = function(x) str_wrap(x, 24), guide = guide_legend(ncol = 2))
# Inches, copied from the original primary panel scripts (not alternative palettes).
# New h/i panels use the dimensions of the analogous Figure 2 PCC panels.
panel_sizes <- list(
 figure2a=c(6,6), figure2b=c(6,6), figure2c=c(6,6),
 figure2d=c(12.5,7), figure2e=c(12.5,7), figure2f=c(6,6), figure2g=c(8,16),
 figure2h=c(7,4), figure2i=c(7,4),
 figure3a=c(7,4), figure3b=c(8,6), figure3c=c(5,6), figure3ac=c(9,6),
 figure3e=c(24,6), figure3f=c(4.5,6),
 figure4a2=c(4,8), figure4a3=c(7.5,10), figure4b=c(16,5), figure4b_svg=c(10,5), figure4b_nonsvg=c(10,5), figure4c=c(5,5),
 figure5a=c(14,4.5))
# Keep manuscript lettering, including both components of Figure 4a.
panel_letter <- function(name) sub("^figure[0-9]+([a-z]).*$", "\\1", name)
save_panel <- function(p,name,d,tag=panel_letter(name)) {
 size <- panel_sizes[[name]]
 if (is.null(size)) stop("Missing output dimensions for ", name)
 if ("Method" %in% names(d)) {
  unknown <- setdiff(unique(as.character(d$Method)), names(method_colours))
  if (length(unknown)) stop("Add model colours for: ", paste(unknown, collapse=", "))
 }
 if ("value" %in% names(d)) {
  groups <- intersect(c("Method","Gene_set","Gene_type","Metric","Dataset","Direction"),names(d))
  counts <- d |> group_by(across(all_of(groups))) |> summarise(n_total=n(),n_finite=sum(is.finite(value)),n_missing=sum(!is.finite(value)),.groups="drop")
  write_csv(counts,file.path(output_dir,paste0(name,"_counts.csv")))
 }
 p <- p + labs(tag = tag) + theme(
  plot.tag = element_text(face = "bold", size = 18, colour = "black"),
  plot.tag.position = "topleft", plot.margin = margin(8, 8, 8, 8))
 ggsave(file.path(output_dir,paste0(name,".pdf")),p,width=size[1],height=size[2],units="in",device=cairo_pdf)
 write_csv(d,file.path(output_dir,paste0(name,"_plot_data.csv")))
 message("Saved ",name," (",nrow(d)," plotted rows)")
 invisible(p) # Return the labeled ggplot for the figure script's patchwork layout.
}
metric_long <- function(d,m) d |> pivot_longer(ends_with(paste0(" ",m)),names_to="Method",values_to="value") |> mutate(Method=ordered(str_remove(Method,paste0(" ",m,"$"))))
rank_subsets <- function(d,second="Non-SVG rank",label="Non-SVGs") bind_rows(lapply(c(20,50),function(n) bind_rows(
 d |> filter(.data[["SVG rank"]]<=n) |> mutate(Gene_type="SVGs",Gene_set=paste("Top",n)),
 d |> filter(.data[[second]]<=n) |> mutate(Gene_type=label,Gene_set=paste("Top",n)))))
# Explicit facet order: SVGs first, then Non-SVGs (or HVGs for SSIM).
order_gene_panels <- function(d) d |> mutate(
 Gene_type = factor(Gene_type, levels=c("SVGs", "Non-SVGs", "HVGs")),
 Gene_set = factor(Gene_set, levels=c("Top 20", "Top 50")))
box_panel <- function(d,y) ggplot(order_gene_panels(d),aes(Gene_set,value,fill=Method))+
 geom_boxplot(position=position_dodge(.8),outlier.size=.8)+
 facet_wrap(~Gene_type,nrow=1)+colours()+labs(x=NULL,y=y)+style()
spatial_points <- function(size=.3) {
 p <- geom_point(size=size,stroke=0)
 if(requireNamespace("ggrastr",quietly=TRUE)) ggrastr::rasterise(p,dpi=300) else p
}

# Xenium image calibration supplied by the user (micrometres per pixel).
xenium_um_per_pixel <- 0.2125
# Translate each tissue/ROI to a local origin and invert image y for plotting.
# Translation preserves physical distances; x and y use the same conversion.
spatial_coordinates_um <- function(d, x, y, groups=character()) {
 stopifnot(all(is.finite(d[[x]])), all(is.finite(d[[y]])))
 d |> group_by(across(all_of(groups))) |>
  mutate(x_um=(.data[[x]]-min(.data[[x]]))*xenium_um_per_pixel,
         y_um=(max(.data[[y]])-.data[[y]])*xenium_um_per_pixel) |> ungroup()
}
# Overlay a vector scale bar inside each tissue; bottom-left by default.
# White outlines surround both the black bar and label; no background box or extra margins.
# Call with fixed facet scales and coord_equal() to retain physical geometry.
add_spatial_scale_bar <- function(p, d, facets, length_um, label, normalized=FALSE, position=c("bottom-left", "top-right")) {
 position <- match.arg(position)
 bars <- d |> group_by(across(all_of(facets))) |>
  summarise(span_x=max(x_um), span_y=max(y_um), .groups="drop")
 stopifnot(all(bars$span_x > 0), all(bars$span_y > 0), length_um > 0,
           all(length_um < .8*bars$span_x))
 bars <- bars |> mutate(
  x_start=.05*span_x, x_end=.05*span_x+length_um,
  y_bar=.07*span_y, y_label=.16*span_y, label=label)
 if (position == "top-right") bars <- bars |> mutate(
  x_start=.95*span_x-length_um, x_end=.95*span_x,
  y_bar=.84*span_y, y_label=.93*span_y)
 # In normalized panels, convert the annotation to the same 0–1 coordinates.
 # The horizontal bar remains calibrated; it does not describe vertical distances.
 if (normalized) bars <- bars |> mutate(
  x_start=x_start/span_x, x_end=x_end/span_x,
  y_bar=y_bar/span_y, y_label=y_label/span_y)
 # A wider white stroke underneath creates a halo around the black bar.
 # Restore a background grid on every spatial map. Physical maps use the
 # scale-bar length as grid spacing; normalized maps use quarter-unit spacing.
 grid_breaks <- if (normalized) seq(0, 1, by=.25) else function(limits) {
  seq(floor(limits[1]/length_um), ceiling(limits[2]/length_um)) * length_um
 }
 p + scale_x_continuous(breaks=grid_breaks, minor_breaks=NULL) +
  scale_y_continuous(breaks=grid_breaks, minor_breaks=NULL) +
  theme(panel.grid.major=element_line(colour="grey75", linewidth=.3),
        panel.grid.minor=element_blank(), panel.ontop=FALSE) +
  geom_segment(data=bars, aes(x=x_start, xend=x_end, y=y_bar, yend=y_bar),
               inherit.aes=FALSE, colour="white", linewidth=1.4,
               lineend="round", show.legend=FALSE) +
  geom_segment(data=bars, aes(x=x_start, xend=x_end, y=y_bar, yend=y_bar),
               inherit.aes=FALSE, colour="black", linewidth=.8,
               lineend="round", show.legend=FALSE) +
  shadowtext::geom_shadowtext(
   data=bars, aes(x=(x_start+x_end)/2, y=y_label, label=label),
   inherit.aes=FALSE, colour="black", bg.colour="white", bg.r=.12,
   size=3.2, show.legend=FALSE)
}
