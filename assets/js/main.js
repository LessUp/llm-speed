/**
 * LLM-Speed GitHub Pages - Main JavaScript
 * Interactive features: theme, navigation, TOC, scroll progress
 */

(function() {
  'use strict';

  // ═════════════════════════════════════════════════════════════════════════════
  // THEME MANAGEMENT
  // ═════════════════════════════════════════════════════════════════════════════

  const ThemeManager = {
    currentTheme: 'auto',

    init() {
      this.loadTheme();
      this.bindEvents();
      this.applyTheme();
    },

    loadTheme() {
      this.currentTheme = localStorage.getItem('theme') || 'auto';
    },

    saveTheme() {
      localStorage.setItem('theme', this.currentTheme);
    },

    applyTheme() {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      const isDark = this.currentTheme === 'dark' || 
                     (this.currentTheme === 'auto' && prefersDark);
      
      document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
    },

    cycleTheme() {
      const themes = ['auto', 'light', 'dark'];
      const currentIndex = themes.indexOf(this.currentTheme);
      this.currentTheme = themes[(currentIndex + 1) % themes.length];
      this.saveTheme();
      this.applyTheme();
      this.updateToggleIcon();
    },

    updateToggleIcon() {
      const toggle = document.getElementById('theme-toggle');
      if (!toggle) return;
      
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      const isDark = this.currentTheme === 'dark' || 
                     (this.currentTheme === 'auto' && prefersDark);
      
      toggle.setAttribute('aria-label', 
        isDark ? 'Switch to light theme' : 'Switch to dark theme'
      );
    },

    bindEvents() {
      const toggle = document.getElementById('theme-toggle');
      if (toggle) {
        toggle.addEventListener('click', () => this.cycleTheme());
      }

      // Listen for system theme changes
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        if (this.currentTheme === 'auto') {
          this.applyTheme();
        }
      });
    }
  };

  // ═════════════════════════════════════════════════════════════════════════════
  // MOBILE NAVIGATION
  // ═════════════════════════════════════════════════════════════════════════════

  const MobileNav = {
    isOpen: false,

    init() {
      this.bindEvents();
    },

    bindEvents() {
      const toggle = document.getElementById('menu-toggle');
      const overlay = document.getElementById('overlay');
      const mobileNav = document.getElementById('mobile-nav');

      if (toggle) {
        toggle.addEventListener('click', () => this.toggle());
      }

      if (overlay) {
        overlay.addEventListener('click', () => this.close());
      }

      // Close on link click
      if (mobileNav) {
        mobileNav.querySelectorAll('a').forEach(link => {
          link.addEventListener('click', () => this.close());
        });
      }

      // Close on escape key
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && this.isOpen) {
          this.close();
        }
      });
    },

    toggle() {
      this.isOpen ? this.close() : this.open();
    },

    open() {
      this.isOpen = true;
      document.getElementById('mobile-nav')?.classList.add('active');
      document.getElementById('overlay')?.classList.add('active');
      document.body.style.overflow = 'hidden';
    },

    close() {
      this.isOpen = false;
      document.getElementById('mobile-nav')?.classList.remove('active');
      document.getElementById('overlay')?.classList.remove('active');
      document.body.style.overflow = '';
    }
  };

  // ═════════════════════════════════════════════════════════════════════════════
  // READING PROGRESS
  // ═════════════════════════════════════════════════════════════════════════════

  const ReadingProgress = {
    init() {
      this.progressBar = document.getElementById('reading-progress');
      if (!this.progressBar) return;
      
      this.bindEvents();
    },

    bindEvents() {
      window.addEventListener('scroll', () => this.update(), { passive: true });
      window.addEventListener('resize', () => this.update(), { passive: true });
    },

    update() {
      const scrollTop = window.scrollY;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const progress = (scrollTop / docHeight) * 100;
      
      this.progressBar.style.width = Math.min(progress, 100) + '%';
    }
  };

  // ═════════════════════════════════════════════════════════════════════════════
  // TABLE OF CONTENTS
  // ═════════════════════════════════════════════════════════════════════════════

  const TableOfContents = {
    headings: [],
    tocLinks: [],

    init() {
      this.generateTOC();
      this.highlightCurrentSection();
      this.bindEvents();
    },

    generateTOC() {
      const content = document.getElementById('page-content');
      const toc = document.getElementById('toc');
      
      if (!content || !toc) return;

      // Find all h2 and h3 headings
      this.headings = Array.from(content.querySelectorAll('h2[id], h3[id]'));
      
      if (this.headings.length === 0) {
        document.getElementById('toc-wrapper')?.classList.add('hidden');
        return;
      }

      const ul = document.createElement('ul');
      
      this.headings.forEach(heading => {
        const li = document.createElement('li');
        const a = document.createElement('a');
        
        a.href = '#' + heading.id;
        a.textContent = heading.textContent.replace('#', '').trim();
        a.dataset.target = heading.id;
        
        if (heading.tagName === 'H3') {
          li.style.paddingLeft = '1rem';
        }
        
        li.appendChild(a);
        ul.appendChild(li);
        
        this.tocLinks.push({ link: a, heading: heading });
      });

      toc.appendChild(ul);
    },

    highlightCurrentSection() {
      if (this.headings.length === 0) return;

      const scrollPos = window.scrollY + 100;
      let currentHeading = null;

      // Find the current heading
      for (let i = this.headings.length - 1; i >= 0; i--) {
        if (this.headings[i].offsetTop <= scrollPos) {
          currentHeading = this.headings[i];
          break;
        }
      }

      // Update active states
      this.tocLinks.forEach(({ link, heading }) => {
        link.classList.remove('active');
        if (heading === currentHeading) {
          link.classList.add('active');
        }
      });
    },

    bindEvents() {
      window.addEventListener('scroll', () => {
        requestAnimationFrame(() => this.highlightCurrentSection());
      }, { passive: true });

      // Smooth scroll for TOC links
      this.tocLinks.forEach(({ link }) => {
        link.addEventListener('click', (e) => {
          e.preventDefault();
          const target = document.getElementById(link.dataset.target);
          if (target) {
            const offset = 80; // Header height + padding
            const top = target.offsetTop - offset;
            window.scrollTo({ top, behavior: 'smooth' });
            history.pushState(null, null, link.href);
          }
        });
      });
    }
  };

  // ═════════════════════════════════════════════════════════════════════════════
  // BACK TO TOP
  // ═════════════════════════════════════════════════════════════════════════════

  const BackToTop = {
    init() {
      this.button = document.getElementById('back-to-top');
      if (!this.button) return;
      
      this.bindEvents();
    },

    bindEvents() {
      window.addEventListener('scroll', () => this.toggle(), { passive: true });
      
      this.button.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    },

    toggle() {
      if (window.scrollY > 500) {
        this.button.classList.add('visible');
      } else {
        this.button.classList.remove('visible');
      }
    }
  };

  // ═════════════════════════════════════════════════════════════════════════════
  // CODE BLOCK ENHANCEMENTS
  // ═════════════════════════════════════════════════════════════════════════════

  const CodeBlocks = {
    init() {
      this.enhanceCodeBlocks();
    },

    enhanceCodeBlocks() {
      document.querySelectorAll('pre code').forEach(block => {
        const pre = block.parentElement;
        const lang = this.detectLanguage(block);
        
        if (lang) {
          pre.setAttribute('data-language', lang);
        }
      });
    },

    detectLanguage(block) {
      const className = block.className;
      const match = className.match(/language-(\w+)/);
      return match ? match[1] : '';
    }
  };

  // ═════════════════════════════════════════════════════════════════════════════
  // SEARCH FUNCTIONALITY
  // ═════════════════════════════════════════════════════════════════════════════

  const Search = {
    searchIndex: [],
    searchContainer: null,
    searchInput: null,
    searchResults: null,

    init() {
      this.searchContainer = document.getElementById('search-container');
      this.searchInput = document.getElementById('search-input');
      this.searchResults = document.getElementById('search-results');

      if (!this.searchInput) return;

      this.bindEvents();
      this.loadSearchIndex();
    },

    bindEvents() {
      // Toggle search on mobile
      const searchToggle = document.getElementById('search-toggle');
      if (searchToggle) {
        searchToggle.addEventListener('click', () => {
          this.searchContainer?.classList.toggle('active');
          if (this.searchContainer?.classList.contains('active')) {
            this.searchInput?.focus();
          }
        });
      }

      // Input handling with debounce
      let debounceTimer;
      this.searchInput.addEventListener('input', (e) => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => this.performSearch(e.target.value), 200);
      });

      // Clear button
      const clearBtn = document.getElementById('search-clear');
      if (clearBtn) {
        clearBtn.addEventListener('click', () => {
          this.searchInput.value = '';
          this.searchInput.focus();
          this.hideResults();
        });
      }

      // Close on escape
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          this.hideResults();
          this.searchContainer?.classList.remove('active');
        }
      });

      // Close on outside click
      document.addEventListener('click', (e) => {
        if (!this.searchContainer?.contains(e.target)) {
          this.hideResults();
        }
      });
    },

    async loadSearchIndex() {
      // Check if search-index.js is loaded
      if (window.searchIndexData) {
        this.searchIndex = window.searchIndexData;
      }
    },

    performSearch(query) {
      if (!query.trim()) {
        this.hideResults();
        return;
      }

      const results = this.search(query);
      this.displayResults(results);
    },

    search(query) {
      const normalizedQuery = query.toLowerCase();
      return this.searchIndex.filter(item => {
        return item.title.toLowerCase().includes(normalizedQuery) ||
               item.content.toLowerCase().includes(normalizedQuery);
      }).slice(0, 10);
    },

    displayResults(results) {
      if (!this.searchResults) return;

      if (results.length === 0) {
        this.searchResults.innerHTML = `
          <div class="search-empty">
            ${this.searchInput?.placeholder?.includes('中文') ? '未找到结果' : 'No results found'}
          </div>
        `;
        this.searchResults.classList.add('active');
        return;
      }

      const baseUrl = document.querySelector('meta[property="og:url"]')?.content 
        ? new URL(document.querySelector('meta[property="og:url"]').content).pathname
        : '';

      this.searchResults.innerHTML = results.map(result => `
        <div class="search-result-item">
          <a href="${result.url}" class="search-result-link">
            <div class="search-result-title">${this.highlightMatch(result.title)}</div>
            <div class="search-result-excerpt">${this.highlightMatch(result.excerpt)}</div>
          </a>
        </div>
      `).join('');

      this.searchResults.classList.add('active');
    },

    highlightMatch(text) {
      const query = this.searchInput?.value?.trim();
      if (!query) return text;
      
      const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
      return text.replace(regex, '<span class="search-result-highlight">$1</span>');
    },

    hideResults() {
      this.searchResults?.classList.remove('active');
    }
  };

  // ═════════════════════════════════════════════════════════════════════════════
  // LANGUAGE SWITCHER
  // ═════════════════════════════════════════════════════════════════════════════

  const LanguageSwitcher = {
    init() {
      this.bindEvents();
    },

    bindEvents() {
      // Close language menu on outside click
      document.addEventListener('click', (e) => {
        const switcher = document.querySelector('.language-switcher');
        if (switcher && !switcher.contains(e.target)) {
          switcher.querySelector('.lang-menu')?.classList.remove('active');
        }
      });
    }
  };

  // ═════════════════════════════════════════════════════════════════════════════
  // ANIMATIONS ON SCROLL
  // ═════════════════════════════════════════════════════════════════════════════

  const ScrollAnimations = {
    init() {
      this.observeElements();
    },

    observeElements() {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('fade-in');
            observer.unobserve(entry.target);
          }
        });
      }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
      });

      document.querySelectorAll('.feature-card, .stat-card').forEach(el => {
        observer.observe(el);
      });
    }
  };

  // ═════════════════════════════════════════════════════════════════════════════
  // INITIALIZE ALL MODULES
  // ═════════════════════════════════════════════════════════════════════════════

  document.addEventListener('DOMContentLoaded', () => {
    ThemeManager.init();
    MobileNav.init();
    ReadingProgress.init();
    TableOfContents.init();
    BackToTop.init();
    CodeBlocks.init();
    Search.init();
    LanguageSwitcher.init();
    ScrollAnimations.init();
  });

})();
