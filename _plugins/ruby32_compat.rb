# Ruby 3.2+ removed Object#tainted?, but liquid 4.0.3 (pinned exactly by the
# github-pages gem) still calls it on every rendered variable. Modern Ruby has
# no taint tracking, so reporting "not tainted" is correct semantics -- it
# mirrors jekyll 4, which dropped the taint check entirely.
# No-op on Ruby <= 3.1, where the real method still exists.
class Object
  def tainted?
    false
  end unless method_defined?(:tainted?)
end