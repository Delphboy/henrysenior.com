# Compatibility shims so this 2019-era jekyll 3.9 / github-pages stack runs
# on modern Ruby. github-pages 232+ needs commonmarker < 1.0 (Ruby < 4.0), so
# we're pinned to the jekyll 3.9 graph on Ruby 4.0 -- these two fixes bring it
# up to date. Both are no-ops on older Rubies.

# 1. Ruby 3.2+ removed Object#tainted?, but liquid 4.0.3 (pinned exactly by
#    github-pages) still calls it on every rendered variable. Modern Ruby has
#    no taint tracking, so "not tainted" is correct semantics -- it mirrors
#    jekyll 4, which dropped the taint check entirely.
class Object
  def tainted?
    false
  end unless method_defined?(:tainted?)
end

# 2. jekyll 3.9's watch mode reads /proc/version via Pathutil#read, which
#    passes a trailing Hash positionally to File.read -- a keyword-args
#    TypeError on Ruby 3.x. jekyll 3.10 fixed this by reading the file
#    directly, so do the same. (Patching Pathutil itself would be broader
#    than needed; this is the only call site.)
module Jekyll
  module Utils
    module Platforms
      private

      def proc_version
        @proc_version ||= begin
          File.read("/proc/version")
        rescue Errno::ENOENT
          nil
        end
      end
    end
  end
end