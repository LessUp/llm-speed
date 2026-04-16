/**
 * Search functionality for LLM-Speed documentation
 * Uses Lunr.js for client-side search
 */

(function() {
  'use strict';

  // Search data will be populated by the build process
  window.searchIndexData = [];

  // Initialize search when DOM is ready
  document.addEventListener('DOMContentLoaded', function() {
    // Load search index if available
    const searchScript = document.createElement('script');
    searchScript.src = (document.querySelector('meta[property="og:url"]')?.content 
      ? new URL(document.querySelector('meta[property="og:url"]').content).pathname 
      : '') + '/assets/js/search-index.js';
    searchScript.async = true;
    searchScript.onerror = function() {
      console.log('Search index not available');
    };
    document.head.appendChild(searchScript);
  });
})();
