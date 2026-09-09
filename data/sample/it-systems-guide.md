# IT Systems Guide

Written by Sam Ogilvie, IT Manager. This is for everyone, not just people who
like computers. If something here does not make sense, that is my fault and I want
to know about it.

**IT support: (503) 555-0142, option 2. Monday-Friday 7 a.m. to 6 p.m. Pacific.**
After hours, if the POS is down and you cannot take payment, call the same number
and it forwards to my cell. Anything less urgent than that can wait until morning.

## The Systems

### BrewPoint (point of sale)

Our register system, built on Square. Every cafe has two iPads, and the kiosks have
one plus a spare in the back.

Log in with your 4-digit code, which your manager sets up on your first day. Never
use someone else's code. When we investigate a drawer discrepancy the first thing
we look at is whose code rang the transaction, and it is not fair to anyone if
codes are shared.

End of day, close out in BrewPoint before you count the drawer. If you count first
the numbers will not line up and you will think you are short when you are not.

### BeanTrack (inventory)

Where we track beans, milk, pastry, cups, everything. Web-based, works in any
browser, and there is an iPad app on the back-of-house tablet.

Counts go in nightly. Beans by the pound, milk by the case, pastry by the unit.
Takes about ten minutes if you do it consistently and forty-five if you skip three
days. Reorder points are already set, so when something turns yellow it means
order it, and red means you are going to run out tomorrow.

Supply tickets also live in BeanTrack. If a piece of equipment is broken, put the
ticket there rather than texting your manager, because that is how it gets to the
service provider with a paper trail.

### Deputy (scheduling)

Schedules, shift swaps, time-off requests, and clock-in. Download the app, it is
better than the website.

Schedules post Wednesday for the following week. Swaps need both people to accept
and then a manager to approve. Time off is 14 days ahead, more if it is a holiday.
Clock in on the store tablet, not your phone, since the phone clock-in is not
location-verified and it will kick back to your manager for approval.

### WiFi

Two networks at every location:

- **WallysGuest** is for customers. The password is on the chalkboard and rotates
  monthly.
- **WallysOps** is for registers, tablets, and the back office. Never give this to
  a customer, and never join a personal device to it.

The kiosks in Boise are on a cellular modem, not fixed broadband, so if the
internet feels slow there that is usually why. BrewPoint holds transactions
offline and syncs when it recovers.

### VPN (corporate staff only)

Portland office folks who need the file server or the finance system from home:
install Cisco Secure Client, log in with your email and the MFA push. If MFA is
not prompting, your phone lost the enrollment and I have to re-enroll you. Cafe
staff do not need VPN for anything.

## Submitting a Ticket

Three ways, in order of preference:

1. BeanTrack, Support tab, New Ticket. Best option because it gets logged.
2. Email helpdesk@wallyscoffee.example. Creates a ticket automatically.
3. Phone, for anything blocking sales right now.

Tell me the location, which device, what you were doing, and what it said. A
photo of the error screen is worth more than a paragraph describing it.

## Troubleshooting

### BrewPoint frozen

This happens most often mid-morning rush, and it is almost always memory.

1. Wait 30 seconds. It sometimes recovers on its own and a hard restart mid-sale
   can orphan the transaction.
2. Force-quit BrewPoint. Swipe up from the bottom, hold, swipe the app away.
   Reopen it. About 80 percent of freezes end here.
3. Still stuck: hold the power and volume-up buttons until you see the Apple logo.
   Give it two minutes to come back and sync.
4. Switch to the second iPad and keep taking payment. Do not stop selling while
   you troubleshoot.
5. If both iPads are down, take cash only, write orders on the pad, and call IT.
   The offline queue will reconcile once it reconnects.

Do not delete and reinstall BrewPoint. You will lose the offline queue and the
device enrollment, and I will have to rebuild it.

### Receipt printer jam

1. Open the lid, pull the paper out toward you, never back through the mechanism.
2. Check for a torn scrap under the cutter. That is the usual culprit.
3. Reload with the paper feeding from underneath the roll, not over the top. Half
   of all jams are a backwards roll.
4. Close the lid until it clicks, press feed twice. If it grinds, power-cycle at
   the switch on the back.
5. Still jammed: BeanTrack ticket. Kiosks keep a spare printer in the back.

### BeanTrack not syncing

Symptoms are counts that will not save, or last night's numbers missing.

1. Confirm you are on WallysOps and not WallysGuest.
2. Hard refresh the browser, Ctrl+Shift+R, or force-quit the iPad app.
3. Check the sync indicator, top right. Amber means queued, red means failed.
4. Amber for over an hour usually means the connection dropped mid-write. Log out
   fully, log back in, and re-enter that day's count.
5. Red, or numbers still missing after a re-login, is a ticket for me. Do not
   re-enter the same count repeatedly, because it can double-post and then the
   inventory looks wrong in the other direction.
