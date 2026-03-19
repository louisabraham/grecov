#' Compute tail probabilities via BFS
#'
#' Low-level function that computes left and right tail probabilities
#' for a multinomial weighted sum using breadth-first search.
#'
#' @param p Numeric vector of multinomial probabilities.
#' @param v Numeric vector of category values.
#' @param s_obs Observed test statistic (the weighted sum).
#' @param n Total count (sum of observations).
#' @param eps Stopping tolerance on unexplored mass (default 1e-4).
#'
#' @return A list with components:
#' \describe{
#'   \item{prob_left}{P(S <= s_obs).}
#'   \item{prob_right}{P(S >= s_obs).}
#'   \item{explored_mass}{Total probability mass visited.}
#'   \item{states_explored}{Number of states explored.}
#' }
#'
#' @examples
#' \dontrun{
#' result <- grecov_tail(
#'   p = c(0.3, 0.7),
#'   v = c(0, 1),
#'   s_obs = 8,
#'   n = 10L
#' )
#' cat(sprintf("P(S <= 8) = %.6f\n", result$prob_left))
#' }
#'
#' @export
grecov_tail <- function(p, v, s_obs, n, eps = 1e-4) {
  result <- grecov_mod$grecov_tail(
    p = as.double(p),
    v = as.double(v),
    s_obs = as.double(s_obs),
    n = as.integer(n),
    eps = eps
  )

  list(
    prob_left = result$prob_left,
    prob_right = result$prob_right,
    explored_mass = result$explored_mass,
    states_explored = as.integer(result$states_explored)
  )
}
